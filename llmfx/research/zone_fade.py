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
from ..domain.types import Candle, SwingType
from ..domain.zones import ZoneTracker


def defenders_weakening(
    swings: list, from_below: bool, bar_index: int
) -> bool:
    """帯の守り手が押し負け始めているか(利用者の言う「ブレイクリスク」)。

        安値切り上げや高値切り下げが起き始めた抵抗帯はブレイクされやすい。
        = その帯の守り手が諦め始めている。

    抵抗帯(上から売る側)なら、直近の確定安値が切り上がっていれば
    買い手が押し上げてきているということ。支持帯ならその鏡像。

    確定済みのスイングだけを見る。未確定を混ぜると先読みになる。
    判定できるだけの本数が無ければ False(見送らない)。
    """
    kind = SwingType.LOW if from_below else SwingType.HIGH
    pivots = [s for s in swings if s.type is kind and s.confirmed_index <= bar_index]
    if len(pivots) < 2:
        return False
    rising = pivots[-1].price > pivots[-2].price
    return rising if from_below else not rising


def _near_nfp(moment, minutes: int) -> bool:
    """米雇用統計の前後 `minutes` 分か。毎月第 1 金曜と決まっている。"""
    from datetime import timedelta

    from ..domain.sessions import _neighbour_months, nfp_time

    window = timedelta(minutes=minutes)
    for year, month in _neighbour_months(moment.year, moment.month):
        if abs(moment - nfp_time(year, month)) <= window:
            return True
    return False


def _atr_series(candles: list[Candle], period: int) -> list[float]:
    """各足までの ATR(Wilder)。上位足で帯を引くとき、下位足の値幅を
    測るのに要る。"""
    out: list[float] = []
    run = 0.0
    prev = candles[0].close if candles else 0.0
    for i, c in enumerate(candles):
        tr = c.high - c.low if i == 0 else max(
            c.high - c.low, abs(c.high - prev), abs(c.low - prev)
        )
        run = tr if i == 0 else (run * (period - 1) + tr) / period
        prev = c.close
        out.append(run)
    return out


def _rolling_extremes(
    candles: list[Candle], window: int | None
) -> tuple[list[float] | None, list[float] | None]:
    """直近 `window` 本の高値の最大・安値の最小を、各足について先に作る。

    単調デックで O(n)。窓を毎回舐めると O(n x window) になり、
    掃引が実用にならない(48 通りで 1 銘柄 1 時間を超えた)。
    """
    if not window:
        return None, None
    from collections import deque

    highs: list[float] = []
    lows: list[float] = []
    hi_q: deque[int] = deque()
    lo_q: deque[int] = deque()
    for i, c in enumerate(candles):
        while hi_q and candles[hi_q[-1]].high <= c.high:
            hi_q.pop()
        hi_q.append(i)
        while lo_q and candles[lo_q[-1]].low >= c.low:
            lo_q.pop()
        lo_q.append(i)
        if hi_q[0] <= i - window:
            hi_q.popleft()
        if lo_q[0] <= i - window:
            lo_q.popleft()
        highs.append(candles[hi_q[0]].high)
        lows.append(candles[lo_q[0]].low)
    return highs, lows


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
    entry_hour: int = 0
    """約定した足の UTC 時。スプレッドが開く時間帯を後から評価するのに使う。"""


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
    higher_minutes: int | None = None,
    require_range: bool = False,
    max_range_atr: float | None = None,
    exit_at_opposite_zone: bool = False,
    blocked_hours_utc: frozenset[int] | None = None,
    nfp_blackout_minutes: int = 0,
    skip_break_risk: bool = False,
    entry_from_range_bars: int | None = None,
    stop_from_range_bars: int | None = None,
    warmup: int = 200,
    refresh_every: int = 50,
) -> list[FadeTrade]:
    """帯へ指値を置き、約定した取引だけを集める。

    `entry_offset_atr` は帯の手前側の縁から **帯の内側へ** どれだけ
    入れて指値を置くか。利用者の言う「少し奥」。0 なら縁ちょうど。
    `stop_buffer_atr` は帯の向こう側の縁から外側へどれだけ離すか。

    `skip_break_risk` は利用者の言う「ブレイクリスク」で見送る。

        安値切り上げや高値切り下げが起き始めた抵抗帯はブレイクされやすい。
        = その帯の守り手が諦め始めている。

    抵抗帯(上から売る)なら、直近の確定安値が切り上がっていたら見送る。
    買い手が押し上げてきている = 守り手が押し負けつつある。
    支持帯なら鏡像で、直近の確定高値が切り下がっていたら見送る。

    利用者は「そもそもエントリーしない」と言っていた。指値を動かして
    避けるのではなく、**その帯を丸ごと touch しない** のが本来の形。

    `entry_from_range_bars` は **指値そのもの** を直近 N 本の最値に置く。
    利用者の本来の指摘はこちら:

        1 つ前の帯の最値ではなく、より広い範囲での最値にすることで、
        よりはみ出しに刈られにくくなる。

    帯の縁の少し内側で待つと、上へ突き抜ける動きの **途中** で約定して
    しまい、そのまま損切りまで持っていかれる。直近 N 本の最値まで
    引き上げると、**そこまで実際に届いたときにしか約定しない**。
    売るなら天井で売る、という形になる。

    最値は **1 本前まで** で作る。その足自身の高値を使うと、
    「その足で最値を更新したから、その値で約定した」という循環になる。

    `stop_from_range_bars` は損切りの基準を帯の縁ではなく直近 N 本の
    最値にする。指値を最値へ置く場合は、損切りはそこから
    `stop_buffer_atr` だけ外側になるので、リスク幅を直接決められる。

    `higher_minutes` を渡すと **帯を上位足で引く**。利用者の指定は
    「エントリーに使う時間軸の 2 つ上位」(M15 なら H1)。閉じた上位足
    しか使わないので先読みにならない。

    `exit_at_opposite_zone` は、**反対側の帯へ届いたらそこで手仕舞う**。
    利用者の説明:

        もしその区画に抵抗帯が二つあれば、自ずとそれらはレンジになる。
        その為、両端からエントリーする必要がある。

    帯 1 本での逆張りは、反対側まで走っても時間切れまで持ち続ける。
    反対側で切れば、そこは同時に **反対向きのエントリー地点** でもあるので、
    往復を刈れるようになる(手仕舞い後は待機が解けるため、同じ足で
    反対側の帯に指値を置ける)。

    注意: 反対側で切ると、**そのまま抜けて走る場合の裾も切る**。
    利用者は「反対側を抜けてそのまま走ったら全部が取り分」と言っている
    ので、ここは掃引して確かめる軸であって、既定では入れない。

    `blocked_hours_utc` はその UTC 時に **建玉を持たない**。

    実勢のスプレッドは時間帯で数倍に開く。とくに NY 17 時のロールオーバー
    (UTC 21-22 時、日本の早朝)は薄く、平常時の数倍になる。固定 pips で
    測ると、この時間帯の取引だけコストを大幅に過小評価する。

    **どの時間が薄いかは板の仕組みから事前に分かる。**成績を見てから
    悪い時間を外すのは選択バイアスだが、ロールオーバーを外すのは
    先読みにならない。

    `nfp_blackout_minutes` は米雇用統計の前後この分数を避ける。毎月第 1
    金曜と決まっているので事前に分かる。**指標は年 100 回程度で、
    毎日あるロールオーバーとは頻度が桁で違う**ため、効きは小さいはず。

    `require_range` は **上下 2 本の帯が揃っているときだけ**建玉を持つ。
    こちらは能力ではなく絞り込み。既定では掛けない。
    利用者の説明:

        抵抗帯を 2 つ探し、そこをレンジとしてその間の往復を刈り取る。

    片側だけで張ると、そこを抜けられたときに一方的に負ける。両側を
    押さえていれば、抜けた側が次のエントリーになる。
    `max_range_atr` は上下の間隔の上限(離れすぎた 2 本を組にしない)。
    """
    from datetime import timedelta

    from ..data.resample import resample_candles

    detector = SwingDetector(
        left=left, right=right, atr_period=atr_period, min_swing_atr=min_swing_atr
    )
    tracker = ZoneTracker(tolerance_atr=tolerance_atr, max_age_bars=max_age_bars)

    # 帯を上位足で引く場合は、閉じた上位足だけを取り込む。
    higher = resample_candles(candles, higher_minutes) if higher_minutes else None
    span = timedelta(minutes=higher_minutes) if higher_minutes else None
    atr_high = _atr_series(higher, atr_period) if higher else None
    atr_low = _atr_series(candles, atr_period) if higher else None
    hi_i = 0
    hi_bar = -1

    # 直近 N 本の最値は、帯に触れるたびに窓を舐め直すと重い。
    # 実測では 48 通りの掃引が 1 銘柄 1 時間を超えた。O(n) で先に作る。
    roll_high, roll_low = _rolling_extremes(candles, stop_from_range_bars)
    entry_high, entry_low = _rolling_extremes(candles, entry_from_range_bars)

    atr_at: list[float] = []
    seen_swings = 0
    trades: list[FadeTrade] = []
    armed: dict[int, bool] = {}
    busy_until = -1
    cached: list = []
    cached_swings = -1
    cached_at = -10**9

    for i, candle in enumerate(candles):
        if higher is not None:
            while hi_i < len(higher) and higher[hi_i].time + span <= candle.time:
                detector.update(higher[hi_i])
                hi_bar = hi_i
                for swing in detector.swings[seen_swings:]:
                    tracker.update(swing, atr=atr_high[hi_i], bar_index=hi_bar)
                seen_swings = len(detector.swings)
                hi_i += 1
            a = atr_low[i]
        else:
            detector.update(candle)
            a = detector.atr or 0.0
            for swing in detector.swings[seen_swings:]:
                tracker.update(swing, atr=a, bar_index=i)
            seen_swings = len(detector.swings)
        atr_at.append(a)

        if i < warmup or a <= 0 or i + max_wait_bars + horizon >= len(candles):
            continue
        if i <= busy_until:
            continue

        # 有効な帯の一覧を毎足作り直すと、蓄積した帯の数に比例して重くなる
        # (600,000 足 x 数千の帯)。スイングが増えたときと、古い帯が落ちる
        # 頃合いだけ作り直す。**新しい帯の反映が遅れる方向にしかずれない**
        # ので、先読みにはならない。
        age_index = hi_bar if higher is not None else i
        if age_index < 0:
            continue
        if seen_swings != cached_swings or i - cached_at >= refresh_every:
            cached = tracker.zones(bar_index=age_index, min_touches=min_touches)
            cached_swings, cached_at = seen_swings, i

        # 上下 2 本の帯が揃っているときだけ触る(利用者の言う「レンジ」)。
        # 片側だけで張ると、そこを抜けられたときに一方的に負ける。
        usable = cached
        if require_range:
            price = candle.close
            above = [z for z in cached if z.price > price]
            below = [z for z in cached if z.price <= price]
            if not above or not below:
                continue
            top = min(above, key=lambda z: z.price)
            bottom = max(below, key=lambda z: z.price)
            if max_range_atr is not None and (top.price - bottom.price) > max_range_atr * a:
                continue
            usable = [top, bottom]

        for zone in usable:
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

            # ブレイクリスク: 守り手が押し負け始めている帯には近づかない。
            if skip_break_risk and defenders_weakening(
                detector.swings, from_below, i
            ):
                continue

            if from_below:
                if entry_high is not None:
                    # 天井で売る。最値は 1 本前まで(循環を避ける)。
                    limit = max(zone.high, entry_high[i - 1])
                else:
                    limit = zone.low + entry_offset_atr * a
                edge = max(zone.high, limit)
                if roll_high is not None:
                    # 帯そのものの縁ではなく、もっと広い範囲の最値を使う。
                    # 帯を作ったスイングの縁だけだと、少しのはみ出しで
                    # 刈られる。利用者の指摘。
                    edge = max(edge, roll_high[i])
                stop = edge + stop_buffer_atr * a
            else:
                if entry_low is not None:
                    limit = min(zone.low, entry_low[i - 1])
                else:
                    limit = zone.high - entry_offset_atr * a
                edge = min(zone.low, limit)
                if roll_low is not None:
                    edge = min(edge, roll_low[i])
                stop = edge - stop_buffer_atr * a
            risk = abs(stop - limit)
            if risk <= 0:
                continue

            # 指値が約定するのは、価格がそこへ **届いた** ときだけ。
            fill_at = None
            for j in range(i, min(i + max_wait_bars, len(candles))):
                c = candles[j]
                if blocked_hours_utc and c.time.hour in blocked_hours_utc:
                    continue
                if nfp_blackout_minutes and _near_nfp(c.time, nfp_blackout_minutes):
                    continue
                if (c.high >= limit) if from_below else (c.low <= limit):
                    fill_at = j
                    break
            if fill_at is None:
                armed[key] = False
                continue

            armed[key] = False

            # 反対側(利益方向)の帯。往復を刈るときの手仕舞い先。
            opposite = None
            if exit_at_opposite_zone:
                other = [z for z in cached if z is not zone
                         and (z.price < limit if from_below else z.price > limit)]
                if other:
                    opposite = (max(other, key=lambda z: z.price) if from_below
                                else min(other, key=lambda z: z.price))

            sign = -1.0 if from_below else 1.0
            # **約定した足そのものから見る。**その足の残りで損切りまで
            # 走ることは普通にある。翌足から数えると、いちばん不利な
            # 場面だけを見逃して成績が良く出る。
            forward = candles[fill_at : fill_at + 1 + horizon]
            if len(forward) < horizon + 1:
                continue

            best = worst = 0.0
            hit_stop = False
            held = horizon
            result = 0.0
            for step, c in enumerate(forward):
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
                if opposite is not None:
                    # 反対側の帯の **手前の縁** で手仕舞う。
                    edge = opposite.high if from_below else opposite.low
                    reached = (c.low <= edge) if from_below else (c.high >= edge)
                    if reached:
                        held = step
                        result = (edge - limit) * sign / risk
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
                    entry_hour=candles[fill_at].time.hour,
                )
            )
            busy_until = fill_at + held   # 同時に 1 建玉。決済したら次を張れる
            break

    return trades
