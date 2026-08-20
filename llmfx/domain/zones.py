"""抵抗帯 — 何度も試された価格水準を数える.

ここまでの実装が入っていたのは **1 回しか触られていないスイング高値** で、
そこを抜けた瞬間に順張りすると、コストをゼロにしても負けた
(22,264 件、-0.071 R、t=-5.33)。押し戻している側がいる。

利用者の見立ては「そこを守っている大口がいる」というもので、
実際の週足には防衛ラインが目に見える形で残っている。ただし
**「160 円に壁がある」と後から言うのは先読み** なので、
その時点までの値動きだけから機械的に数える必要がある。

数え方は素直に:

    確定済みのスイングを、ATR に比例した幅でまとめる
    同じ束に入った本数 = その水準が試された回数

「触られた回数が多い帯ほど跳ね返しが強い」のか
「試されるほど守り手の注文が減って脆い」のかは、どちらも同じくらい
有名な言い伝えで、対立していない。跳ね返れば一段目で取り、
抜ければ二段目で取る。ここが数えるのは回数だけで、良し悪しは決めない。

先読み対策として、`update()` に渡すバー位置より後に確定するスイングは
受け付けない(`confirmed_index > bar_index` なら無視する)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Swing, SwingType


@dataclass
class Zone:
    """同じ水準に集まったスイングの束."""

    low: float
    high: float
    touches: list[Swing] = field(default_factory=list)
    penetrations: list[float] = field(default_factory=list)
    """毎回の接触が、それまでの水準をどれだけ超えたか(ATR 倍)。

    利用者の指摘:「大事なのは防衛ラインを見ることではなく、
    **そこで誰が頑張っているか**を見ること」。守り手が持ちこたえて
    いるなら、試すたびに食い込みは浅くなる。押し負けているなら深くなる。

    比較の基準は **それ以前の接触だけ** から作る。今回の値を混ぜると
    自分自身と比べることになり、常に 0 付近になってしまう。
    最初の接触には比較対象が無いので記録しない(要素数は count - 1)。
    """

    @property
    def price(self) -> float:
        """帯の代表値。触れられた点の平均。"""
        return sum(s.price for s in self.touches) / len(self.touches)

    @property
    def count(self) -> int:
        return len(self.touches)

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def first_index(self) -> int:
        return min(s.index for s in self.touches)

    @property
    def last_index(self) -> int:
        return max(s.index for s in self.touches)

    @property
    def last_confirmed_index(self) -> int:
        return max(s.confirmed_index for s in self.touches)

    @property
    def sides(self) -> set[SwingType]:
        """高値として試されたか、安値として試されたか、両方か。

        一度抜けた抵抗は支持に変わる、と言われる。両側から試された帯は
        その形になっているので、区別できるようにしておく。
        """
        return {s.type for s in self.touches}

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    @property
    def defence(self) -> float | None:
        """守り手が持ちこたえているか。負なら守勢が強い、正なら押されている。

        直近の食い込みと、それ以前の食い込みの平均との差。
        2 回目以降が 2 つ以上ないと判定できないので None を返す。
        """
        if len(self.penetrations) < 2:
            return None
        recent = self.penetrations[-1]
        earlier = self.penetrations[:-1]
        return recent - sum(earlier) / len(earlier)

    @property
    def holding(self) -> bool | None:
        """食い込みが浅くなってきているか(守り手が勝っているか)。"""
        trend = self.defence
        return None if trend is None else trend < 0


class ZoneTracker:
    """確定したスイングを受け取り、価格帯にまとめていく。

    `tolerance_atr` は「同じ水準とみなす幅」を ATR の何倍で測るか。
    銘柄ごとに値幅が違うので、絶対値ではなく ATR 比で持つ。
    """

    def __init__(
        self,
        tolerance_atr: float = 0.5,
        max_age_bars: int | None = None,
    ) -> None:
        if tolerance_atr <= 0:
            raise ValueError("tolerance_atr は正の数である必要があります")
        if max_age_bars is not None and max_age_bars <= 0:
            raise ValueError("max_age_bars は正の数か None です")
        self.tolerance_atr = tolerance_atr
        self.max_age_bars = max_age_bars
        self._zones: list[Zone] = []
        self._seen: set[tuple[int, str]] = set()

    def update(self, swing: Swing, atr: float, bar_index: int) -> Zone | None:
        """スイングを 1 つ取り込む。属した帯を返す。

        まだ確定していないスイング(`confirmed_index > bar_index`)は
        受け付けない。ここを緩めると先読みになる。
        """
        if swing.confirmed_index > bar_index:
            return None
        key = (swing.index, swing.type.value)
        if key in self._seen:
            return None
        if atr <= 0:
            return None
        self._seen.add(key)

        tolerance = atr * self.tolerance_atr
        for zone in self._zones:
            if abs(swing.price - zone.price) <= tolerance:
                # 食い込みは「それ以前の接触だけ」で作った水準と比べる。
                # 今回の値を混ぜると自分自身との比較になってしまう。
                level = zone.price
                depth = (
                    swing.price - level
                    if swing.type is SwingType.HIGH
                    else level - swing.price
                )
                zone.penetrations.append(depth / atr)
                zone.touches.append(swing)
                zone.low = min(zone.low, swing.price - tolerance / 2)
                zone.high = max(zone.high, swing.price + tolerance / 2)
                return zone

        zone = Zone(
            low=swing.price - tolerance / 2,
            high=swing.price + tolerance / 2,
            touches=[swing],
        )
        self._zones.append(zone)
        return zone

    def zones(self, bar_index: int, min_touches: int = 1) -> list[Zone]:
        """いま参照してよい帯。古すぎるものは落とす。"""
        out = []
        for zone in self._zones:
            if zone.count < min_touches:
                continue
            if zone.last_confirmed_index > bar_index:
                continue
            if (
                self.max_age_bars is not None
                and bar_index - zone.last_index > self.max_age_bars
            ):
                continue
            out.append(zone)
        return out

    def nearest(
        self, price: float, bar_index: int, min_touches: int = 2
    ) -> Zone | None:
        """その価格にいちばん近い帯。無ければ None。"""
        candidates = self.zones(bar_index, min_touches=min_touches)
        if not candidates:
            return None
        return min(candidates, key=lambda z: abs(z.price - price))

    def touching(
        self, high: float, low: float, bar_index: int, min_touches: int = 2
    ) -> list[Zone]:
        """その足の高安が重なった帯をすべて返す。"""
        return [
            z
            for z in self.zones(bar_index, min_touches=min_touches)
            if z.low <= high and z.high >= low
        ]
