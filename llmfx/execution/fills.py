"""約定モデルと建玉管理.

バックテストとペーパー取引で **同じコード** を使うためにここへ切り出す。
両者でロジックが分岐すると、バックテストで検証した挙動が実運用で
再現しないという最悪の形で裏切られる。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig
from ..domain.types import Candle, ExitReason, Position, Side


@dataclass(frozen=True)
class FillModel:
    """mid 足に対するスプレッドとスリッページのモデル。

    - 成行・逆指値は不利方向へずらす
    - 指値(利確)は水準どおりに約定するものとする
    """

    pip_size: float
    spread_pips: float
    slippage_pips: float

    @classmethod
    def from_config(cls, config: AppConfig) -> "FillModel":
        return cls(
            pip_size=config.instrument.pip_size,
            spread_pips=config.execution.spread_pips,
            slippage_pips=config.execution.slippage_pips,
        )

    @property
    def cost_per_side(self) -> float:
        return (self.spread_pips / 2.0 + self.slippage_pips) * self.pip_size

    def entry(self, side: Side, price: float) -> float:
        cost = self.cost_per_side
        return price + cost if side is Side.LONG else price - cost

    def exit(self, side: Side, price: float, market: bool) -> float:
        if not market:
            return price
        cost = self.cost_per_side
        return price - cost if side is Side.LONG else price + cost


def evaluate_exit(
    position: Position,
    candle: Candle,
    fills: FillModel,
    bar_index: int,
    max_bars_in_trade: int,
) -> tuple[float, ExitReason] | None:
    """このバーで手仕舞いになるかを判定する。

    同一足で損切りと利確の両方に触れた場合は、足の中の順序が分からない
    以上、必ず損切りが先に約定したものとして扱う。
    """
    long = position.side is Side.LONG

    # 窓開けで損切り水準を飛び越えた場合は始値で約定する。
    gapped = (
        candle.open <= position.stop_loss if long else candle.open >= position.stop_loss
    )
    if gapped:
        return fills.exit(position.side, candle.open, market=True), ExitReason.STOP_LOSS

    hit_stop = candle.low <= position.stop_loss if long else candle.high >= position.stop_loss
    hit_target = (
        candle.high >= position.take_profit if long else candle.low <= position.take_profit
    )

    if hit_stop:
        reason = (
            ExitReason.TRAILING_STOP if position.moved_to_break_even else ExitReason.STOP_LOSS
        )
        return fills.exit(position.side, position.stop_loss, market=True), reason
    if hit_target:
        return fills.exit(position.side, position.take_profit, market=False), ExitReason.TAKE_PROFIT

    if max_bars_in_trade > 0 and bar_index - position.entry_index >= max_bars_in_trade:
        return fills.exit(position.side, candle.close, market=True), ExitReason.TIME_STOP
    return None


def update_stop(
    position: Position,
    candle: Candle,
    config: AppConfig,
    structure_anchor: float | None,
    atr: float,
) -> None:
    """建値移動と構造への追従で損切りを引き上げる(下げることはしない)。

    `structure_anchor` はロングなら直近スイング安値、ショートなら直近
    スイング高値。None の場合は構造トレーリングを行わない。
    """
    cfg = config.execution
    long = position.side is Side.LONG

    excursion = (
        candle.high - position.entry_price if long else position.entry_price - candle.low
    )
    adverse = (
        position.entry_price - candle.low if long else candle.high - position.entry_price
    )
    position.max_favorable_excursion = max(position.max_favorable_excursion, excursion)
    position.max_adverse_excursion = max(position.max_adverse_excursion, adverse)

    if (
        cfg.break_even_at_r is not None
        and not position.moved_to_break_even
        and position.initial_risk_per_unit > 0
        and excursion >= cfg.break_even_at_r * position.initial_risk_per_unit
    ):
        improves = (
            position.entry_price > position.stop_loss
            if long
            else position.entry_price < position.stop_loss
        )
        if improves:
            position.stop_loss = position.entry_price
            position.moved_to_break_even = True

    if not cfg.trail_to_structure or structure_anchor is None:
        return

    buffer = atr * config.entry.stop_buffer_atr
    candidate = structure_anchor - buffer if long else structure_anchor + buffer
    improves = candidate > position.stop_loss if long else candidate < position.stop_loss
    # 現在値を追い越す損切りは即時約定になってしまうので置かない。
    safe = candidate < candle.close if long else candidate > candle.close
    if improves and safe:
        position.stop_loss = candidate
        position.moved_to_break_even = True
