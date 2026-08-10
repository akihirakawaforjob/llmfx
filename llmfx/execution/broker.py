"""ブローカー抽象と実装.

PaperBroker  : 完全シミュレーション。バックテストと同じ約定モデルを使う。
OandaBroker  : OANDA デモ口座へ実際に発注する。SL/TP は約定と同時に
               サーバ側へ置くため、こちらのプロセスが落ちても建玉は保護される。

⚠️ OandaBroker はこの開発環境から API 資格情報にアクセスできないため、
   実接続テストが未実施です。初回はデモ口座かつ最小ロットで確認してください。
"""

from __future__ import annotations

import logging
from typing import Protocol

from ..config import AppConfig
from ..domain.risk import position_size
from ..domain.types import Candle, ExitReason, Position, Side, Signal, Trade
from .fills import FillModel, evaluate_exit

logger = logging.getLogger(__name__)


class Broker(Protocol):
    """実行ループが必要とする最小のブローカー機能。"""

    def equity(self) -> float: ...

    def position(self) -> Position | None: ...

    def submit(self, signal: Signal, candle: Candle, bar_index: int) -> Position | None: ...

    def on_candle(self, candle: Candle, bar_index: int) -> Trade | None: ...

    def close_position(self, candle: Candle, bar_index: int, reason: ExitReason) -> Trade | None: ...


# ----------------------------------------------------------------------
class PaperBroker:
    """シミュレーション執行。実弾を出さずにロジック全体を検証する。"""

    def __init__(self, config: AppConfig, starting_equity: float | None = None) -> None:
        self.config = config
        self.fills = FillModel.from_config(config)
        self._equity = starting_equity or config.risk.initial_equity
        self._position: Position | None = None
        self.trades: list[Trade] = []

    # -- Broker ---------------------------------------------------------
    def equity(self) -> float:
        return self._equity

    def position(self) -> Position | None:
        return self._position

    def submit(self, signal: Signal, candle: Candle, bar_index: int) -> Position | None:
        if self._position is not None:
            return None

        entry_price = self.fills.entry(signal.side, candle.close)
        if signal.side is Side.LONG and entry_price <= signal.stop_loss:
            logger.warning("約定価格が損切りを割り込むため発注を見送ります")
            return None
        if signal.side is Side.SHORT and entry_price >= signal.stop_loss:
            logger.warning("約定価格が損切りを超えるため発注を見送ります")
            return None

        risk_fraction = min(
            self.config.risk.risk_per_trade, self.config.risk.max_risk_per_trade
        )
        sizing_equity = (
            self._equity if self.config.risk.compounding else self.config.risk.initial_equity
        )
        units, risk_amount = position_size(
            equity=sizing_equity,
            risk_fraction=risk_fraction,
            entry_price=entry_price,
            stop_price=signal.stop_loss,
            quote_to_account_rate=self.config.instrument.quote_to_account_rate,
        )
        if units <= 0:
            logger.warning("資金が不足しているため発注できません(必要数量が 1 単位未満)")
            return None

        self._position = Position(
            signal=signal,
            side=signal.side,
            units=units,
            entry_price=entry_price,
            entry_time=candle.time,
            entry_index=bar_index,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            initial_risk_per_unit=abs(entry_price - signal.stop_loss),
            risk_amount=risk_amount,
        )
        return self._position

    def on_candle(self, candle: Candle, bar_index: int) -> Trade | None:
        if self._position is None:
            return None
        exit_info = evaluate_exit(
            position=self._position,
            candle=candle,
            fills=self.fills,
            bar_index=bar_index,
            max_bars_in_trade=self.config.execution.max_bars_in_trade,
        )
        if exit_info is None:
            return None
        price, reason = exit_info
        return self._finalize(candle, bar_index, price, reason)

    def close_position(
        self, candle: Candle, bar_index: int, reason: ExitReason
    ) -> Trade | None:
        if self._position is None:
            return None
        price = self.fills.exit(self._position.side, candle.close, market=True)
        return self._finalize(candle, bar_index, price, reason)

    # -- internals ------------------------------------------------------
    def _finalize(
        self, candle: Candle, bar_index: int, exit_price: float, reason: ExitReason
    ) -> Trade:
        position = self._position
        assert position is not None
        rate = self.config.instrument.quote_to_account_rate
        gross = (
            (exit_price - position.entry_price) * position.side.sign * position.units * rate
        )
        commission = self.config.execution.commission_per_unit * position.units
        pnl = gross - commission
        self._equity += pnl

        trade = Trade(
            side=position.side,
            units=position.units,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_time=candle.time,
            exit_price=exit_price,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            initial_risk_per_unit=position.initial_risk_per_unit,
            risk_amount=position.risk_amount,
            pnl=pnl,
            r_multiple=pnl / position.risk_amount if position.risk_amount > 0 else 0.0,
            exit_reason=reason,
            bars_held=bar_index - position.entry_index,
            equity_after=self._equity,
            rr_at_entry=position.signal.rr,
            target_source=position.signal.target_source,
            structure=position.signal.structure,
            max_favorable_excursion=position.max_favorable_excursion,
            max_adverse_excursion=position.max_adverse_excursion,
            entry_note=position.entry_note,
        )
        self.trades.append(trade)
        self._position = None
        return trade


# ----------------------------------------------------------------------
class OandaBroker:
    """OANDA デモ口座への発注アダプタ.

    損益は口座サマリの実現損益(`pl`)の差分から算出する。約定の細部を
    トランザクション API から復元するより単純で、口座の実際の数字と
    ずれないため。
    """

    def __init__(self, config: AppConfig, client, allow_live: bool = False) -> None:
        self.config = config
        self.client = client
        self.allow_live = allow_live
        self._position: Position | None = None
        self._realized_pl_at_entry: float = 0.0
        self.trades: list[Trade] = []

        if getattr(client, "environment", "practice") == "live" and not allow_live:
            raise RuntimeError(
                "本番口座での実行は allow_live=True の明示が必要です。"
                "まずデモ口座で検証してください。"
            )

    # -- Broker ---------------------------------------------------------
    def equity(self) -> float:
        return self.client.account_equity()

    def position(self) -> Position | None:
        return self._position

    def submit(self, signal: Signal, candle: Candle, bar_index: int) -> Position | None:
        if self._position is not None:
            return None

        equity = self.equity()
        risk_fraction = min(
            self.config.risk.risk_per_trade, self.config.risk.max_risk_per_trade
        )
        units, risk_amount = position_size(
            equity=equity,
            risk_fraction=risk_fraction,
            entry_price=candle.close,
            stop_price=signal.stop_loss,
            quote_to_account_rate=self.config.instrument.quote_to_account_rate,
        )
        if units <= 0:
            logger.warning("必要数量が 1 単位未満のため発注しません")
            return None

        signed_units = units if signal.side is Side.LONG else -units
        precision = _price_precision(self.config.instrument.pip_size)
        response = self.client.place_market_order(
            instrument=self.config.instrument.symbol,
            units=signed_units,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            price_precision=precision,
            client_tag="llmfx",
        )
        fill = response.get("orderFillTransaction")
        if fill is None:
            logger.error("注文が約定しませんでした: %s", response)
            return None

        entry_price = float(fill["price"])
        self._realized_pl_at_entry = self._realized_pl()
        self._position = Position(
            signal=signal,
            side=signal.side,
            units=abs(float(fill["units"])),
            entry_price=entry_price,
            entry_time=candle.time,
            entry_index=bar_index,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            initial_risk_per_unit=abs(entry_price - signal.stop_loss),
            risk_amount=risk_amount,
        )
        return self._position

    def on_candle(self, candle: Candle, bar_index: int) -> Trade | None:
        """建玉がブローカー側で決済されていれば、その事実を取り込む。"""
        if self._position is None:
            return None
        symbol = self.config.instrument.symbol
        still_open = any(p.get("instrument") == symbol for p in self.client.open_positions())
        if still_open:
            return None

        reason = self._infer_exit_reason(candle)
        return self._finalize(candle, bar_index, candle.close, reason)

    def close_position(
        self, candle: Candle, bar_index: int, reason: ExitReason
    ) -> Trade | None:
        if self._position is None:
            return None
        self.client.close_position(self.config.instrument.symbol, self._position.side.value)
        return self._finalize(candle, bar_index, candle.close, reason)

    # -- internals ------------------------------------------------------
    def _realized_pl(self) -> float:
        summary = self.client.account_summary()
        return float(summary["account"].get("pl", 0.0))

    def _infer_exit_reason(self, candle: Candle) -> ExitReason:
        position = self._position
        assert position is not None
        if position.side is Side.LONG:
            if candle.low <= position.stop_loss:
                return ExitReason.STOP_LOSS
            if candle.high >= position.take_profit:
                return ExitReason.TAKE_PROFIT
        else:
            if candle.high >= position.stop_loss:
                return ExitReason.STOP_LOSS
            if candle.low <= position.take_profit:
                return ExitReason.TAKE_PROFIT
        return ExitReason.MANUAL

    def _finalize(
        self, candle: Candle, bar_index: int, exit_price: float, reason: ExitReason
    ) -> Trade:
        position = self._position
        assert position is not None
        pnl = self._realized_pl() - self._realized_pl_at_entry
        equity_after = self.equity()

        trade = Trade(
            side=position.side,
            units=position.units,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_time=candle.time,
            exit_price=exit_price,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            initial_risk_per_unit=position.initial_risk_per_unit,
            risk_amount=position.risk_amount,
            pnl=pnl,
            r_multiple=pnl / position.risk_amount if position.risk_amount > 0 else 0.0,
            exit_reason=reason,
            bars_held=bar_index - position.entry_index,
            equity_after=equity_after,
            rr_at_entry=position.signal.rr,
            target_source=position.signal.target_source,
            structure=position.signal.structure,
            max_favorable_excursion=position.max_favorable_excursion,
            max_adverse_excursion=position.max_adverse_excursion,
            entry_note=position.entry_note,
        )
        self.trades.append(trade)
        self._position = None
        return trade


def _price_precision(pip_size: float) -> int:
    """pip 幅から価格の小数桁数を決める(0.01 -> 3 桁、0.0001 -> 5 桁)。"""
    if pip_size >= 0.01:
        return 3
    return 5
