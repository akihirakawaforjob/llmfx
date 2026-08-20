"""約定モデルと建玉管理.

バックテストとペーパー取引で **同じコード** を使うためにここへ切り出す。
両者でロジックが分岐すると、バックテストで検証した挙動が実運用で
再現しないという最悪の形で裏切られる。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

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
    spread_bps: float = 0.0

    @classmethod
    def from_config(cls, config: AppConfig) -> "FillModel":
        return cls(
            pip_size=config.instrument.pip_size,
            spread_pips=config.execution.spread_pips,
            slippage_pips=config.execution.slippage_pips,
            spread_bps=config.execution.spread_bps,
        )

    def cost_per_side(self, price: float) -> float:
        """片道あたりの不利分。固定 pips と価格比例 bps の合計。"""
        fixed = (self.spread_pips / 2.0 + self.slippage_pips) * self.pip_size
        proportional = abs(price) * (self.spread_bps / 2.0) / 10_000.0
        return fixed + proportional

    def entry(self, side: Side, price: float) -> float:
        cost = self.cost_per_side(price)
        return price + cost if side is Side.LONG else price - cost

    def exit(self, side: Side, price: float, market: bool) -> float:
        if not market:
            return price
        cost = self.cost_per_side(price)
        return price - cost if side is Side.LONG else price + cost


@dataclass(frozen=True)
class TradeCosts:
    """1 トレードにかかる、価格差以外の費用."""

    commission: float
    holding: float

    @property
    def total(self) -> float:
        return self.commission + self.holding


def rollovers_crossed(entry_time: datetime, exit_time: datetime, hour_utc: int) -> int:
    """建玉が日跨ぎの区切りを何回越えたか。

    GMOコインは日本時間 6:00(= UTC 21:00)をまたいで建玉を持つと
    レバレッジ手数料が発生する。エントリー直後に決済すれば 0 回。
    """
    if exit_time <= entry_time:
        return 0
    boundary = entry_time.replace(hour=hour_utc, minute=0, second=0, microsecond=0)
    if boundary <= entry_time:
        boundary += timedelta(days=1)
    if boundary >= exit_time:
        return 0
    return int((exit_time - boundary).total_seconds() // 86_400) + 1


def trade_costs(
    config: AppConfig,
    *,
    units: float,
    entry_price: float,
    exit_price: float,
    entry_time: datetime,
    exit_time: datetime,
) -> TradeCosts:
    """手数料と建玉管理料を計算する。

    バックテストとペーパー取引で同じ数字を使うため、ここに一本化している。

    建玉管理料の評価額には入建値と決済値の平均を使う。実際の GMOコインは
    各営業日の終値で評価するため厳密には一致しないが、保有中の平均的な
    評価額の近似としてはこれで足りる(手数料率自体が 0.04%/日 と小さい)。
    """
    cfg = config.execution
    notional_entry = abs(entry_price) * units
    notional_exit = abs(exit_price) * units

    commission = cfg.commission_per_unit * units
    commission += (notional_entry + notional_exit) * cfg.commission_bps / 10_000.0

    holding = 0.0
    if cfg.daily_holding_cost_bps:
        nights = rollovers_crossed(
            entry_time, exit_time, cfg.holding_cost_rollover_hour_utc
        )
        average_notional = (notional_entry + notional_exit) / 2.0
        holding = average_notional * nights * cfg.daily_holding_cost_bps / 10_000.0

    rate = config.instrument.quote_to_account_rate
    return TradeCosts(commission=commission * rate, holding=holding * rate)


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


def evaluate_partial_exit(
    position: Position,
    candle: Candle,
    config: AppConfig,
) -> float | None:
    """このバーで一部利確が成立するなら、その約定価格を返す。

    水準は指値なのでスリッページを乗せない。ただし **同じ足で損切りにも
    触れていたら成立させない**。足の中の順序が分からない以上、損切りが
    先に約定した扱いにするのがこのプロジェクトの決まりで、
    そこを緩めると「都合の良い順序」を仮定したことになる。
    """
    level_r = config.execution.partial_exit_at_r
    if level_r is None or position.scaled_out or position.initial_risk_per_unit <= 0:
        return None

    long = position.side is Side.LONG
    distance = position.initial_risk_per_unit * level_r
    level = position.entry_price + distance if long else position.entry_price - distance

    reached = candle.high >= level if long else candle.low <= level
    if not reached:
        return None

    # 損切りに触れた足では成立させない(窓開けを含む)。
    touched_stop = (
        candle.low <= position.stop_loss or candle.open <= position.stop_loss
        if long
        else candle.high >= position.stop_loss or candle.open >= position.stop_loss
    )
    if touched_stop:
        return None
    return level


def apply_partial_exit(
    position: Position,
    price: float,
    config: AppConfig,
) -> float:
    """一部を利確し、確定した損益を建玉へ積む。返り値は確定した損益。

    残玉の損切りを建値へ移すかは `break_even_after_partial` で決める。
    """
    cfg = config.execution
    units = position.units * cfg.partial_exit_fraction
    move = (
        price - position.entry_price
        if position.side is Side.LONG
        else position.entry_price - price
    )
    pnl = move * units * config.instrument.quote_to_account_rate

    position.units -= units
    position.scaled_units += units
    position.realized_pnl += pnl
    position.scaled_out = True

    if cfg.break_even_after_partial:
        improves = (
            position.entry_price > position.stop_loss
            if position.side is Side.LONG
            else position.entry_price < position.stop_loss
        )
        if improves:
            position.stop_loss = position.entry_price
            position.moved_to_break_even = True
    return pnl


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
