"""利確目標の決定.

リスクリワードフィルタを意味のあるものにするための要。目標を
「損切り幅 x 2」で機械的に置くと RR は常に 2.0 になり、フィルタが
何も選別しなくなる。そこで目標は相場側の客観的な水準から決め、
その結果として得られた RR を判定に使う。

利用できる戦略:
  structure     : 建値より先にある「最も近い」スイング水準(次の抵抗/支持)
  trend_origin  : 反転させられたトレンドの起点(その波動全体の戻し目標)
  measured_move : 転換直前の波の値幅を建値から投影
  atr           : ATR の定数倍(最後の保険)

「水準を出せた最初の戦略」を採用する。RR が足りるまで戦略を渡り歩く
ことはしない(それは目標の後付けであり、フィルタの意味を失わせる)。

なお、損切りを「転換前の極値」に置く以上リスク幅は直前の波の全長に
なるため、structure(最も近い壁)や measured_move(波 1 本分)では
RR が構造的に 1 前後に張り付き、RR>=2 をほぼ通過できない。
ダウ理論の解釈としては trend_origin(トレンド全体の戻しを狙う)の方が
損切り幅と整合する。詳細は README の「RR フィルタの幾何学」を参照。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import EntryConfig
from .types import Side, Swing, SwingType


@dataclass(frozen=True)
class TargetResolution:
    price: float
    source: str


def resolve_target(
    side: Side,
    entry: float,
    risk_per_unit: float,
    swings: list[Swing],
    atr: float,
    config: EntryConfig,
) -> TargetResolution | None:
    """設定された順に利確水準を探し、最初に見つかったものを返す。"""
    for strategy in config.target_strategies:
        resolver = _RESOLVERS.get(strategy)
        if resolver is None:
            continue
        price = resolver(side, entry, risk_per_unit, swings, atr, config)
        if price is None:
            continue
        # 建値と逆方向・同値の目標は無効。
        if (side is Side.LONG and price <= entry) or (
            side is Side.SHORT and price >= entry
        ):
            continue
        return TargetResolution(price=price, source=strategy)
    return None


def _structure_target(
    side: Side,
    entry: float,
    risk_per_unit: float,
    swings: list[Swing],
    atr: float,
    config: EntryConfig,
) -> float | None:
    """建値の先にある最も近いスイング水準を「次の壁」として使う。"""
    min_distance = atr * config.min_target_distance_atr
    recent = swings[-config.structure_lookback_swings :] if swings else []

    if side is Side.LONG:
        candidates = [
            s.price
            for s in recent
            if s.type is SwingType.HIGH and s.price >= entry + min_distance
        ]
        return min(candidates) if candidates else None

    candidates = [
        s.price
        for s in recent
        if s.type is SwingType.LOW and s.price <= entry - min_distance
    ]
    return max(candidates) if candidates else None


def _trend_origin_target(
    side: Side,
    entry: float,
    risk_per_unit: float,
    swings: list[Swing],
    atr: float,
    config: EntryConfig,
) -> float | None:
    """反転させられたトレンドの起点を目標にする。

    下降トレンドが終わったなら、価格はその下降が始まった地点へ向けて
    戻りうる、というダウ理論的な読み。損切りが波 1 本分の幅を取る以上、
    目標も波 1 本より大きい水準に置かないとリスクリワードが釣り合わない。
    """
    min_distance = atr * config.min_target_distance_atr
    recent = swings[-config.structure_lookback_swings :] if swings else []
    if not recent:
        return None

    if side is Side.LONG:
        candidates = [
            s.price
            for s in recent
            if s.type is SwingType.HIGH and s.price >= entry + min_distance
        ]
        return max(candidates) if candidates else None

    candidates = [
        s.price
        for s in recent
        if s.type is SwingType.LOW and s.price <= entry - min_distance
    ]
    return min(candidates) if candidates else None


def _measured_move_target(
    side: Side,
    entry: float,
    risk_per_unit: float,
    swings: list[Swing],
    atr: float,
    config: EntryConfig,
) -> float | None:
    """転換直前の高値-安値の値幅を、建値から同じだけ投影する。"""
    last_high = _last_of(swings, SwingType.HIGH)
    last_low = _last_of(swings, SwingType.LOW)
    if last_high is None or last_low is None:
        return None
    leg = abs(last_high.price - last_low.price)
    if leg <= 0:
        return None
    move = leg * config.measured_move_mult
    return entry + move if side is Side.LONG else entry - move


def _atr_target(
    side: Side,
    entry: float,
    risk_per_unit: float,
    swings: list[Swing],
    atr: float,
    config: EntryConfig,
) -> float | None:
    if atr <= 0:
        return None
    move = atr * config.atr_target_mult
    return entry + move if side is Side.LONG else entry - move


def _fixed_r_target(
    side: Side,
    entry: float,
    risk_per_unit: float,
    swings: list[Swing],
    atr: float,
    config: EntryConfig,
) -> float | None:
    """損切り幅 x min_rr。RR フィルタは常に通るので検証用途にとどめる。"""
    if risk_per_unit <= 0:
        return None
    move = risk_per_unit * config.min_rr
    return entry + move if side is Side.LONG else entry - move


def _last_of(swings: list[Swing], swing_type: SwingType) -> Swing | None:
    for swing in reversed(swings):
        if swing.type is swing_type:
            return swing
    return None


_RESOLVERS = {
    "structure": _structure_target,
    "trend_origin": _trend_origin_target,
    "measured_move": _measured_move_target,
    "atr": _atr_target,
    "fixed_r": _fixed_r_target,
}
