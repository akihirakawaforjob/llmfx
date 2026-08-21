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


THRESHOLDS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
"""順序を記録する ATR 倍のしきい値。損切り・利確の候補水準。"""


@dataclass
class TouchEvent:
    """帯に触れた 1 回分の記録."""

    bar_index: int
    """帯に触れた足。"""
    decision_index: int
    """弾かれた / 抜けたが決まった足。成果はこの足の終値から測る。"""
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
    reaction: str
    """帯に触れた後、`confirm` 本のうちに何を示したか。
    **帯に来た時点では方向を決めない。**

      弾かれた  帯の手前側へ戻って引けた  -> 跳ね返る側に付く
      抜けた    帯の向こう側で引けた      -> 抜けた側に付く
      中        帯の中で引けた            -> まだ分からない

    利用者の説明:「抵抗帯は抜けようが抜けまいが、相場に従うだけ」。
    帯は方向の予測ではなく **判断の場所** で、どちらを向いたかは
    触れた足が教える。ここを混ぜて平均すると、跳ね返りと突破が
    打ち消し合ってきっかり 0 になる(実際そうなった)。
    """
    follow_move: float
    """触れた足が示した向きへ、その後どれだけ動いたか(ATR 倍)。"""
    follow_max: float
    """同じ向きへの最大到達(ATR 倍)。"""
    follow_adverse: float
    """同じ向きに付いた場合の最大逆行(ATR 倍)。損切り幅の目安。"""
    first_favourable: tuple[int, ...] = ()
    """各しきい値へ **最初に届いた** 足(判定足からの本数)。届かなければ -1。

    しきい値は `THRESHOLDS`(ATR 倍)。最大到達だけを見ても、損切りと
    利確のどちらが先だったかは分からない。順序を残しておかないと、
    「両方に触れた足は損切り扱い」という決まりが効きすぎて、
    どんな組み合わせも全部負けに見える(実際そうなった)。
    """
    first_adverse: tuple[int, ...] = ()
    """各しきい値へ最初に逆行した足。届かなければ -1。"""
    zone_width_atr: float = 0.0
    """帯の幅を、測っている足の ATR で割ったもの。

    損切りは帯の外に置くので、これがそのままリスク幅の下限になる。
    上位足で入ると大きく、下位足で入ると小さい。**同じ値動きでも
    取れる R 倍数がここで決まる。**"""


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


def _build_event(
    series: list[Candle],
    i: int,
    zone: Zone,
    atr: float,
    horizon: int,
    confirm: int,
) -> TouchEvent | None:
    """帯へ触れた 1 件を、成果まで含めて組み立てる。

    単一足版と上位足版で **必ず同じ処理を通す** ためにここへ出した。
    二重に書いていたせいで、しきい値の順序を片方にだけ足してしまい、
    2 時間半かけた観測を丸ごと捨てることになった。約定ロジックを
    二重に書かないのと同じ理由で、観測の処理も一本化する。
    """
    # どちら側から来たかは **触れる前** の位置で決める。触れた足の終値で
    # 決めると、突き抜けて上で引けた足が「上から来た」判定になり、
    # 突破が「弾かれた」に化ける(実際に化けていた)。
    from_below = series[i - 1].close < zone.price

    # 帯に来た時点では方向を決めない。`confirm` 本のうちにどちらへ
    # 引けたかで決める。1 本で判定すると突破がほとんど拾えない
    # (合成データで 851 件中 2 件しか出なかった)。
    decision = None
    reaction = "中"
    for j in range(i, min(i + confirm, len(series))):
        c = series[j]
        if from_below:
            if c.close > zone.high:
                decision, reaction = j, "抜けた"
                break
            if c.close < zone.low:
                decision, reaction = j, "弾かれた"
                break
        else:
            if c.close < zone.low:
                decision, reaction = j, "抜けた"
                break
            if c.close > zone.high:
                decision, reaction = j, "弾かれた"
                break
    if decision is None:
        decision, reaction = min(i + confirm - 1, len(series) - 1), "中"

    forward = series[decision + 1 : decision + 1 + horizon]
    if len(forward) < horizon:
        return None

    start = series[decision].close
    sign = -1.0 if from_below else 1.0
    moves = [(c.close - start) * sign / atr for c in forward]
    highs = [(c.high - start) * sign / atr for c in forward]
    lows = [(c.low - start) * sign / atr for c in forward]
    broke = any(
        c.close > zone.high if from_below else c.close < zone.low for c in forward
    )

    # 示された向きへ付いた場合の成果。「中」は跳ね返り側を仮置き。
    follow_sign = -1.0 if reaction == "抜けた" else 1.0
    fhi = [v * follow_sign for v in highs]
    flo = [v * follow_sign for v in lows]

    # しきい値ごとに「最初に届いた足」を残す。最大到達だけでは損切りと
    # 利確のどちらが先か分からず、「両方に触れたら損切り」の決まりが
    # 効きすぎて、どの組み合わせも全部負けに見える。
    first_fav = [-1] * len(THRESHOLDS)
    first_adv = [-1] * len(THRESHOLDS)
    for step, (hi_v, lo_v) in enumerate(zip(fhi, flo)):
        up = max(hi_v, lo_v)
        dn = -min(hi_v, lo_v)
        for k, level in enumerate(THRESHOLDS):
            if first_fav[k] < 0 and up >= level:
                first_fav[k] = step
            if first_adv[k] < 0 and dn >= level:
                first_adv[k] = step

    return TouchEvent(
        bar_index=i,
        decision_index=decision,
        zone_price=zone.price,
        touches=zone.count,
        defence=zone.defence,
        from_below=from_below,
        atr=atr,
        fade_move=moves[-1],
        fade_max=max(max(highs), max(lows)),
        break_max=-min(min(highs), min(lows)),
        broke=broke,
        reaction=reaction,
        follow_move=moves[-1] * follow_sign,
        follow_max=max(max(fhi), max(flo)),
        follow_adverse=-min(min(fhi), min(flo)),
        first_favourable=tuple(first_fav),
        first_adverse=tuple(first_adv),
        zone_width_atr=zone.width / atr,
    )


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
    one_per_bar: bool = False,
    cooldown: int = 0,
    confirm: int = 3,
) -> list[TouchEvent]:
    """帯への接触をすべて拾い、その後 `horizon` 本の値動きを測る。

    同じ帯に何本も張り付いている間は 1 回と数える。いったん帯から
    `rearm_atr` 倍だけ離れたら、次の接触を数えられるようにする。

    標本の重なりについて。1 本の足が複数の帯に触れることがあり、
    近い時刻の事象は同じ値動きを共有する。**重なった標本を独立として
    数えると t 値が大きく出すぎる。**

      one_per_bar  1 本の足からは最も近い帯の 1 件だけを採る
      cooldown     直前の事象からこの本数が経つまで、新しい事象を採らない
                   (`horizon` と同じ値にすれば、成果の窓が重ならない)
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

        if i < warmup or a <= 0 or i + confirm + horizon >= len(candles):
            continue

        # 窓が重なる事象は記録しない。ここを緩めると t 値が嘘をつく。
        # ただし「触れた」こと自体は数えるので、待機の状態は普段どおり動かす。
        # そうしないと、同じ接触の途中の足が後から 1 回目として記録される。
        cooling = bool(cooldown > 0 and events and i - events[-1].bar_index < cooldown)

        found_here: list[TouchEvent] = []
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
            if cooling:
                continue

            # どちら側から来たかは **触れる前** の位置で決める。触れた足の
            # 終値で決めると、突き抜けて上で引けた足が「上から来た」判定に
            # なり、突破が「弾かれた」に化ける(実際に化けていた)。
            event = _build_event(candles, i, zone, a, horizon, confirm)
            if event is None:
                continue
            found_here.append(event)

        if not found_here:
            continue
        if one_per_bar:
            # 同じ足の複数の帯は同じ値動きを共有する。最も近い 1 つに絞る。
            found_here = [
                min(found_here, key=lambda e: abs(e.zone_price - candle.close))
            ]
        events.extend(found_here)
    return events


def collect_touches_mtf(
    higher: list[Candle],
    lower: list[Candle],
    *,
    higher_minutes: int,
    left: int = 3,
    right: int = 3,
    atr_period: int = 14,
    min_swing_atr: float = 0.6,
    tolerance_atr: float = 0.5,
    max_age_bars: int | None = 500,
    min_touches: int = 2,
    horizon: int = 48,
    rearm_atr: float = 1.0,
    warmup: int = 500,
    one_per_bar: bool = True,
    cooldown: int = 0,
    confirm: int = 3,
) -> list[TouchEvent]:
    """帯は上位足で見つけ、接触と成果は下位足で測る。

    利用者の指摘:「跳ね返りを刈るなら、なるべく小さい足で刈る方がいい。
    上の方の足で見たところでエントリーポイントがわからん」。

    影響は見やすさに留まらない。上位足の終値で入ると、その時点で帯から
    離れているぶんだけ損切りが遠くなり、**同じ値動きでも取れる R 倍数が
    小さくなる。**下位足なら帯の際で入れる。

    先読み対策: 上位足のバーは **閉じてから** しか使わない。時刻 T の
    上位足は T + higher_minutes になって初めて参照できる。
    """
    from datetime import timedelta

    span = timedelta(minutes=higher_minutes)
    detector = SwingDetector(
        left=left, right=right, atr_period=atr_period, min_swing_atr=min_swing_atr
    )
    tracker = ZoneTracker(tolerance_atr=tolerance_atr, max_age_bars=max_age_bars)
    atr_low = _atr_series(lower, atr_period)
    atr_high = _atr_series(higher, atr_period)

    events: list[TouchEvent] = []
    armed: dict[int, bool] = {}
    hi_i = 0          # 次に取り込む上位足
    seen_swings = 0
    hi_bar = -1       # 取り込み済みの上位足の本数 - 1(帯の年齢に使う)

    for i, candle in enumerate(lower):
        # 閉じた上位足だけを取り込む。
        while hi_i < len(higher) and higher[hi_i].time + span <= candle.time:
            detector.update(higher[hi_i])
            hi_bar = hi_i
            for swing in detector.swings[seen_swings:]:
                tracker.update(swing, atr=atr_high[hi_i], bar_index=hi_bar)
            seen_swings = len(detector.swings)
            hi_i += 1

        a = atr_low[i]
        if i < warmup or a <= 0 or hi_bar < 0:
            continue
        if i + confirm + horizon >= len(lower):
            continue

        cooling = bool(cooldown > 0 and events and i - events[-1].bar_index < cooldown)
        found_here: list[TouchEvent] = []

        for zone in tracker.zones(bar_index=hi_bar, min_touches=min_touches):
            key = id(zone)
            overlaps = zone.low <= candle.high and zone.high >= candle.low
            if not overlaps:
                away = min(abs(candle.close - zone.low), abs(candle.close - zone.high))
                if away > a * rearm_atr:
                    armed[key] = True
                continue
            if not armed.get(key, True):
                continue
            armed[key] = False
            if cooling:
                continue

            event = _build_event(lower, i, zone, a, horizon, confirm)
            if event is None:
                continue
            found_here.append(event)

        if not found_here:
            continue
        if one_per_bar:
            found_here = [
                min(found_here, key=lambda e: abs(e.zone_price - candle.close))
            ]
        events.extend(found_here)
    return events


@dataclass
class Bucket:
    """まとめた 1 群."""

    label: str
    moves: list[float] = field(default_factory=list)
    fade_max: list[float] = field(default_factory=list)
    break_max: list[float] = field(default_factory=list)
    broke: list[bool] = field(default_factory=list)
    follow: list[float] = field(default_factory=list)
    follow_max: list[float] = field(default_factory=list)
    follow_adverse: list[float] = field(default_factory=list)

    def add(self, e: "TouchEvent") -> None:
        self.moves.append(e.fade_move)
        self.fade_max.append(e.fade_max)
        self.break_max.append(e.break_max)
        self.broke.append(e.broke)
        self.follow.append(e.follow_move)
        self.follow_max.append(e.follow_max)
        self.follow_adverse.append(e.follow_adverse)

    @property
    def mean_follow(self) -> float:
        return sum(self.follow) / len(self.follow) if self.follow else 0.0

    @property
    def median_follow(self) -> float:
        return median(self.follow) if self.follow else 0.0

    @property
    def median_follow_max(self) -> float:
        return median(self.follow_max) if self.follow_max else 0.0

    @property
    def median_follow_adverse(self) -> float:
        return median(self.follow_adverse) if self.follow_adverse else 0.0

    def follow_tstat(self) -> float:
        n = len(self.follow)
        if n < 2:
            return 0.0
        mean = self.mean_follow
        var = sum((v - mean) ** 2 for v in self.follow) / (n - 1)
        return mean / (var**0.5 / n**0.5) if var > 0 else 0.0

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
                b.add(e)
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
        target.add(e)
    return [b for b in (holding, losing, unknown) if b.count]


def bucket_by_reaction(events: list[TouchEvent]) -> list[Bucket]:
    """触れた足が何を示したかでまとめる。

    帯に来た時点では方向を決めず、示された向きに付く。混ぜて平均すると
    跳ね返りと突破が打ち消し合って 0 になるので、必ず分けて見ること。
    """
    order = ["弾かれた", "中", "抜けた"]
    buckets = {k: Bucket(k) for k in order}
    for e in events:
        buckets[e.reaction].add(e)
    return [buckets[k] for k in order if buckets[k].count]
