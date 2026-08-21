"""抵抗帯へ指値を置いて待つ取引を、逆選択まで含めて測る.

利用者が実際にやっていた形:

    抵抗帯の少し奥(スプレッド対策)に予め指値を置いておく
    価格がそこへ届けば約定、届かなければ何も起きない
    損切りは帯の外側
    利確は明示しない(相場に従う)

**指値は「届いたときだけ約定する」ため、都合の良い場面を取りこぼす。**
価格が帯へ少し触れて反転する場面(いちばん美味しい形)では約定せず、
勢いよく突き抜ける場面(いちばん不利な形)では必ず約定する。この偏りを
逆選択という。成行を前提にした集計はこれを見落とし、コストの節約分が
そのまま残ると誤解させる。

ここでは指値の約定を素直に模擬する。届いた足でだけ建玉を持ち、
届かなければ事象そのものを作らない。よって残った標本には
最初から逆選択が織り込まれている。

利用者は約定を待つ間に指値を動かしていたが、それは再現しない。
「どう動かしたか」を後から決めると、値動きを見てから決めたことになる。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.swings import SwingDetector
from ..domain.types import Candle
from ..domain.zones import ZoneTracker


@dataclass
class FadeTrade:
    """帯へ置いた指値が約定してからの成果."""

    bar_index: int
    zone_price: float
    zone_width_atr: float
    touches: int
    atr: float
    from_below: bool
    """True なら下から来た(= 抵抗帯で売る)。"""
    entry: float
    stop: float
    risk_atr: float
    """損切りまでの幅(ATR 倍)。R の分母。"""
    r_multiple: float
    """観測期間の終わりまで持った場合の成果。"""
    max_favourable_r: float
    max_adverse_r: float
    hit_stop: bool
    bars_held: int


def collect_fade_trades(
    candles: list[Candle],
    *,
    left: int = 3,
    right: int = 3,
    atr_period: int = 14,
    min_swing_atr: float = 0.6,
    tolerance_atr: float = 0.5,
    max_age_bars: int | None = 2000,
    min_touches: int = 2,
    entry_offset_atr: float = 0.1,
    stop_buffer_atr: float = 0.2,
    max_wait_bars: int = 12,
    horizon: int = 24,
    max_zone_width_atr: float | None = None,
    warmup: int = 200,
    refresh_every: int = 50,
) -> list[FadeTrade]:
    """帯へ指値を置き、約定した取引だけを集める。

    `entry_offset_atr` は帯の手前側の縁から **帯の内側へ** どれだけ
    入れて指値を置くか。利用者の言う「少し奥」。0 なら縁ちょうど。
    `stop_buffer_atr` は帯の向こう側の縁から外側へどれだけ離すか。
    """
    detector = SwingDetector(
        left=left, right=right, atr_period=atr_period, min_swing_atr=min_swing_atr
    )
    tracker = ZoneTracker(tolerance_atr=tolerance_atr, max_age_bars=max_age_bars)

    atr_at: list[float] = []
    seen_swings = 0
    trades: list[FadeTrade] = []
    armed: dict[int, bool] = {}
    busy_until = -1
    cached: list = []
    cached_swings = -1
    cached_at = -10**9

    for i, candle in enumerate(candles):
        detector.update(candle)
        a = detector.atr or 0.0
        atr_at.append(a)
        for swing in detector.swings[seen_swings:]:
            tracker.update(swing, atr=a, bar_index=i)
        seen_swings = len(detector.swings)

        if i < warmup or a <= 0 or i + max_wait_bars + horizon >= len(candles):
            continue
        if i <= busy_until:
            continue

        # 有効な帯の一覧を毎足作り直すと、蓄積した帯の数に比例して重くなる
        # (600,000 足 x 数千の帯)。スイングが増えたときと、古い帯が落ちる
        # 頃合いだけ作り直す。**新しい帯の反映が遅れる方向にしかずれない**
        # ので、先読みにはならない。
        if seen_swings != cached_swings or i - cached_at >= refresh_every:
            cached = tracker.zones(bar_index=i, min_touches=min_touches)
            cached_swings, cached_at = seen_swings, i

        for zone in cached:
            width = zone.width / a
            if max_zone_width_atr is not None and width > max_zone_width_atr:
                continue
            key = id(zone)

            # 帯から十分離れたら、次の待ち伏せを許す。
            if not (zone.low <= candle.high and zone.high >= candle.low):
                away = min(abs(candle.close - zone.low), abs(candle.close - zone.high))
                if away > a:
                    armed[key] = True
                continue
            if not armed.get(key, True):
                continue

            from_below = candles[i - 1].close < zone.price
            if from_below:
                limit = zone.low + entry_offset_atr * a     # 抵抗帯で売る
                stop = zone.high + stop_buffer_atr * a
            else:
                limit = zone.high - entry_offset_atr * a    # 支持帯で買う
                stop = zone.low - stop_buffer_atr * a
            risk = abs(stop - limit)
            if risk <= 0:
                continue

            # 指値が約定するのは、価格がそこへ **届いた** ときだけ。
            fill_at = None
            for j in range(i, min(i + max_wait_bars, len(candles))):
                c = candles[j]
                if (c.high >= limit) if from_below else (c.low <= limit):
                    fill_at = j
                    break
            if fill_at is None:
                armed[key] = False
                continue

            armed[key] = False
            sign = -1.0 if from_below else 1.0
            forward = candles[fill_at + 1 : fill_at + 1 + horizon]
            if len(forward) < horizon:
                continue

            best = worst = 0.0
            hit_stop = False
            held = horizon
            result = 0.0
            for step, c in enumerate(forward, start=1):
                fav = max((c.high - limit) * sign, (c.low - limit) * sign)
                adv = -min((c.high - limit) * sign, (c.low - limit) * sign)
                best = max(best, fav)
                worst = max(worst, adv)
                # 損切りは高安で判定する。同じ足で有利にも動いていても、
                # 順序が分からない以上こちらを先に見る。
                touched = (c.high >= stop) if from_below else (c.low <= stop)
                if touched:
                    hit_stop = True
                    held = step
                    result = -1.0
                    break
            if not hit_stop:
                result = (forward[-1].close - limit) * sign / risk

            trades.append(
                FadeTrade(
                    bar_index=i,
                    zone_price=zone.price,
                    zone_width_atr=width,
                    touches=zone.count,
                    atr=a,
                    from_below=from_below,
                    entry=limit,
                    stop=stop,
                    risk_atr=risk / a,
                    r_multiple=result,
                    max_favourable_r=best / risk,
                    max_adverse_r=worst / risk,
                    hit_stop=hit_stop,
                    bars_held=held,
                )
            )
            busy_until = fill_at + held   # 同時に 1 建玉だけ
            break

    return trades
