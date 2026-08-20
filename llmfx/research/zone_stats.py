"""抵抗帯に触れた後、値動きがどうなるかを数える.

利用者の見立て:

  抵抗帯では、負け側の大口が資金を投入して押し戻そうとする。
  それに乗る。押し負けて帯が破れたら、諦めた側の損切りに乗る。

これを検証可能な観測に翻訳すると 2 つになる:

  1 試された回数が多い帯ほど、跳ね返りは強いか
  2 食い込みが浅くなっている帯(守り手が持ちこたえている)ほど、
    跳ね返りは強いか。深くなっている帯は抜けやすいか

「回数が多いほど跳ね返す」と「試されるほど脆い」はどちらも同じくらい
有名な言い伝えで、対立していない。跳ね返れば一段目、抜ければ二段目で
取るため。ここは良し悪しを決めず、関係があるかだけを数える。

**戦略ではないので取引しない。**事象を数えるので、標本は取引の桁を
はるかに超える(ダウ転換は 20 年 28 銘柄で 642 件しか無かった)。

先読み対策:
  - 帯は確定済みスイングだけから作る(`ZoneTracker` が守る)
  - 触れた判定はその足の高安。その後の値動きは **翌足以降** だけを見る
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

from ..domain.swings import SwingDetector
from ..domain.types import Candle, SwingType
from ..domain.zones import ZoneTracker


@dataclass
class TouchEvent:
    """帯に触れた 1 回分の記録."""

    bar_index: int
    zone_price: float
    touches: int
    """それまでに何回試されたか(この回を含む)。"""
    defence: float | None
    """食い込みの推移。負なら守り手が持ちこたえている。"""
    from_below: bool
    """下から来て抵抗として試したか(True)、上から来て支持として試したか。"""
    atr: float
    fade_move: float
    """跳ね返り方向へどれだけ動いたか(ATR 倍)。正なら跳ね返った。"""
    fade_max: float
    """跳ね返り方向への最大到達(ATR 倍)。取れた可能性の上限。"""
    break_max: float
    """突破方向への最大到達(ATR 倍)。逆張りした場合の最大逆行。"""
    broke: bool
    """観測期間のうちに帯の向こう側へ抜けたか。"""


def _atr_series(candles: list[Candle], period: int) -> list[float]:
    trs = [candles[0].high - candles[0].low]
    for i in range(1, len(candles)):
        prev = candles[i - 1].close
        c = candles[i]
        trs.append(max(c.high - c.low, abs(c.high - prev), abs(c.low - prev)))
    out, run = [], 0.0
    for i, tr in enumerate(trs):
        run = tr if i == 0 else (run * (period - 1) + tr) / period
        out.append(run)
    return out


def collect_touches(
    candles: list[Candle],
    *,
    left: int = 3,
    right: int = 3,
    atr_period: int = 14,
    min_swing_atr: float = 0.6,
    tolerance_atr: float = 0.5,
    max_age_bars: int | None = 2000,
    min_touches: int = 2,
    horizon: int = 24,
    rearm_atr: float = 1.0,
    warmup: int = 200,
) -> list[TouchEvent]:
    """帯への接触をすべて拾い、その後 `horizon` 本の値動きを測る。

    同じ帯に何本も張り付いている間は 1 回と数える。いったん帯から
    `rearm_atr` 倍だけ離れたら、次の接触を数えられるようにする。
    """
    detector = SwingDetector(
        left=left, right=right, atr_period=atr_period, min_swing_atr=min_swing_atr
    )
    tracker = ZoneTracker(tolerance_atr=tolerance_atr, max_age_bars=max_age_bars)
    atr = _atr_series(candles, atr_period)
    events: list[TouchEvent] = []
    armed: dict[int, bool] = {}
    seen_swings = 0

    for i, candle in enumerate(candles):
        detector.update(candle)
        a = atr[i]
        # 確定したスイングを帯へ積む。detector.swings は確定順に伸びる。
        for swing in detector.swings[seen_swings:]:
            tracker.update(swing, atr=a, bar_index=i)
        seen_swings = len(detector.swings)

        if i < warmup or a <= 0 or i + horizon >= len(candles):
            continue

        for zone in tracker.zones(bar_index=i, min_touches=min_touches):
            key = id(zone)
            overlaps = zone.low <= candle.high and zone.high >= candle.low
            if not overlaps:
                # 帯から十分離れたら、次の接触を数えられるようにする。
                away = min(abs(candle.close - zone.low), abs(candle.close - zone.high))
                if away > a * rearm_atr:
                    armed[key] = True
                continue
            if not armed.get(key, True):
                continue
            armed[key] = False

            from_below = candle.close < zone.price
            forward = candles[i + 1 : i + 1 + horizon]
            if not forward:
                continue
            start = candle.close
            # 跳ね返り方向 = 帯へ来た向きの逆
            sign = -1.0 if from_below else 1.0
            moves = [(c.close - start) * sign / a for c in forward]
            highs = [(c.high - start) * sign / a for c in forward]
            lows = [(c.low - start) * sign / a for c in forward]
            fade_max = max(max(highs), max(lows))
            break_max = -min(min(highs), min(lows))
            broke = any(
                c.close > zone.high if from_below else c.close < zone.low
                for c in forward
            )
            events.append(
                TouchEvent(
                    bar_index=i,
                    zone_price=zone.price,
                    touches=zone.count,
                    defence=zone.defence,
                    from_below=from_below,
                    atr=a,
                    fade_move=moves[-1],
                    fade_max=fade_max,
                    break_max=break_max,
                    broke=broke,
                )
            )
    return events


@dataclass
class Bucket:
    """まとめた 1 群."""

    label: str
    moves: list[float] = field(default_factory=list)
    fade_max: list[float] = field(default_factory=list)
    break_max: list[float] = field(default_factory=list)
    broke: list[bool] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.moves)

    @property
    def mean_move(self) -> float:
        return sum(self.moves) / len(self.moves) if self.moves else 0.0

    @property
    def median_move(self) -> float:
        return median(self.moves) if self.moves else 0.0

    @property
    def break_rate(self) -> float:
        return sum(self.broke) / len(self.broke) if self.broke else 0.0

    @property
    def median_fade_max(self) -> float:
        return median(self.fade_max) if self.fade_max else 0.0

    @property
    def median_break_max(self) -> float:
        return median(self.break_max) if self.break_max else 0.0

    def tstat(self) -> float:
        n = len(self.moves)
        if n < 2:
            return 0.0
        mean = self.mean_move
        var = sum((v - mean) ** 2 for v in self.moves) / (n - 1)
        if var <= 0:
            return 0.0
        return mean / (var**0.5 / n**0.5)


def bucket_by_touches(events: list[TouchEvent], edges=(2, 3, 4, 6, 10)) -> list[Bucket]:
    """試された回数でまとめる。"""
    buckets: list[Bucket] = []
    bounds = list(edges) + [10**9]
    for lo, hi in zip(bounds, bounds[1:]):
        label = f"{lo}回" if hi - lo == 1 else (f"{lo}回以上" if hi > 10**8 else f"{lo}〜{hi-1}回")
        b = Bucket(label)
        for e in events:
            if lo <= e.touches < hi:
                b.moves.append(e.fade_move)
                b.fade_max.append(e.fade_max)
                b.break_max.append(e.break_max)
                b.broke.append(e.broke)
        if b.count:
            buckets.append(b)
    return buckets


def bucket_by_defence(events: list[TouchEvent]) -> list[Bucket]:
    """守り手が持ちこたえているかでまとめる。"""
    holding = Bucket("食い込みが浅くなっている(守勢が優勢)")
    losing = Bucket("食い込みが深くなっている(押し負け)")
    unknown = Bucket("判定できない(接触が少ない)")
    for e in events:
        target = unknown if e.defence is None else (holding if e.defence < 0 else losing)
        target.moves.append(e.fade_move)
        target.fade_max.append(e.fade_max)
        target.break_max.append(e.break_max)
        target.broke.append(e.broke)
    return [b for b in (holding, losing, unknown) if b.count]
