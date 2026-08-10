"""イベント駆動バックテスト.

先読み(look-ahead bias)を持ち込まないための規律:

  - シグナルはバー確定後の終値でのみ判定する
  - 既定の約定は「翌足の始値」。シグナル足の終値で約定させたい場合は
    execution.entry_mode: close にするが、これは楽観的な想定になる
  - 同一足内で損切りと利確の両方に触れた場合は、必ず損切りが先に約定
    したものとして扱う(足の中の順序は分からないため、悪い方を採る)
  - 逆指値の約定にはスリッページを乗せ、指値の約定には乗せない

スプレッドは mid 足に対する往復コストとしてモデル化する。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from ..config import AppConfig
from ..domain.risk import RiskManager, position_size
from ..domain.strategy import DowReversalStrategy
from ..domain.types import (
    Candle,
    ExitReason,
    Position,
    Side,
    Signal,
    SwingType,
    Trade,
)
from ..execution.fills import FillModel, evaluate_exit, update_stop


class EntryGate(Protocol):
    """LLM などによるエントリー拒否権。承認/拒否と理由を返す。"""

    def evaluate(self, signal: Signal, context: dict) -> dict: ...


class TradeJournalist(Protocol):
    """エントリー時の所感と決済時の振り返りを書く。"""

    def entry_note(self, signal: Signal, context: dict) -> dict | None: ...

    def exit_note(self, trade: Trade, context: dict) -> dict | None: ...


@dataclass
class EquityPoint:
    time: datetime
    equity: float
    """含み損益を含む時価評価。ドローダウン計算はこちらを使う。"""
    realized_equity: float


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[EquityPoint]
    config: AppConfig
    signals_generated: int
    signals_taken: int
    rejections: dict[str, int]
    gate_rejections: int
    halt_reason: str | None
    bars: int
    start_time: datetime | None
    end_time: datetime | None

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1].equity if self.equity_curve else self.config.risk.initial_equity


class BacktestEngine:
    def __init__(
        self,
        config: AppConfig,
        gate: EntryGate | None = None,
        journalist: TradeJournalist | None = None,
    ) -> None:
        self.config = config
        self.gate = gate
        self.journalist = journalist

    # ------------------------------------------------------------------
    def run(self, candles: list[Candle]) -> BacktestResult:
        cfg = self.config
        strategy = DowReversalStrategy(cfg)
        risk_manager = RiskManager(
            initial_equity=cfg.risk.initial_equity,
            max_daily_loss=cfg.risk.max_daily_loss,
            max_drawdown_stop=cfg.risk.max_drawdown_stop,
        )

        equity = cfg.risk.initial_equity
        position: Position | None = None
        pending: Signal | None = None
        pending_gate: dict | None = None
        pending_note: dict | None = None

        trades: list[Trade] = []
        curve: list[EquityPoint] = []
        signals_generated = 0
        signals_taken = 0
        gate_rejections = 0

        fills = FillModel.from_config(cfg)

        for index, candle in enumerate(candles):
            # 1) 前バーで確定した発注を、このバーの始値で約定させる。
            if pending is not None and position is None:
                position = self._open_position(
                    pending, candle, equity, fills, index, pending_gate, pending_note
                )
                pending = None
                pending_gate = None
                pending_note = None

            # 2) 保有ポジションの損切り・利確をこのバーの高安で判定する。
            if position is not None:
                exit_info = evaluate_exit(
                    position=position,
                    candle=candle,
                    fills=fills,
                    bar_index=index,
                    max_bars_in_trade=cfg.execution.max_bars_in_trade,
                )
                if exit_info is not None:
                    trade, equity = self._close_position(
                        position, equity, candle, index, *exit_info
                    )
                    trades.append(trade)
                    position = None
                    if self.journalist is not None:
                        note = self.journalist.exit_note(
                            trade, self._context(strategy, candle, index, equity)
                        )
                        trade.exit_note = note

            # 3) 確定足を戦略へ流し込む(スイング更新とダウ転換の検出)。
            signal = strategy.update(candle)

            # 4) 建値移動・トレーリング。構造更新後に行う。
            if position is not None:
                self._manage_stop(position, strategy, candle)

            # 5) 逆方向のダウ転換が出たら手仕舞う。
            #    RR フィルタで落ちた転換でも、構造が崩れた事実は同じなので
            #    シグナルではなく生の転換イベントを見る。
            flip_event = strategy.last_event
            if (
                position is not None
                and cfg.execution.exit_on_structure_flip
                and flip_event is not None
                and flip_event.bar_index == index
                and flip_event.side is not position.side
            ):
                trade, equity = self._close_position(
                    position,
                    equity,
                    candle,
                    index,
                    fills.exit(position.side, candle.close, market=True),
                    ExitReason.STRUCTURE_FLIP,
                )
                trades.append(trade)
                position = None
                if self.journalist is not None:
                    trade.exit_note = self.journalist.exit_note(
                        trade, self._context(strategy, candle, index, equity)
                    )

            # 6) 時価評価とリスク制限の更新。
            mark = self._mark_to_market(position, candle, equity)
            risk_manager.on_bar(mark, candle.time.date())
            curve.append(
                EquityPoint(time=candle.time, equity=mark, realized_equity=equity)
            )

            if risk_manager.halted and position is not None:
                trade, equity = self._close_position(
                    position,
                    equity,
                    candle,
                    index,
                    fills.exit(position.side, candle.close, market=True),
                    ExitReason.RISK_KILL_SWITCH,
                )
                trades.append(trade)
                position = None

            # 7) 新規シグナルの受け付け。
            if signal is None:
                continue
            signals_generated += 1

            if index < cfg.backtest.warmup_bars:
                continue
            if position is not None or pending is not None:
                continue
            allowed, _reason = risk_manager.can_open(mark)
            if not allowed:
                continue

            context = self._context(strategy, candle, index, equity)
            gate_decision: dict | None = None
            if self.gate is not None:
                gate_decision = self.gate.evaluate(signal, context)
                if gate_decision is not None and not gate_decision.get("approve", True):
                    gate_rejections += 1
                    continue

            entry_note = None
            if self.journalist is not None:
                entry_note = self.journalist.entry_note(signal, context)

            signals_taken += 1
            if cfg.execution.entry_mode == "close":
                position = self._open_position(
                    signal, candle, equity, fills, index, gate_decision, entry_note,
                    fill_price_source="close",
                )
                if position is None:
                    signals_taken -= 1
            else:
                pending = signal
                pending_gate = gate_decision
                pending_note = entry_note

        # データ終端で建玉が残っていれば手仕舞う。
        if position is not None and candles:
            last = candles[-1]
            trade, equity = self._close_position(
                position,
                equity,
                last,
                len(candles) - 1,
                fills.exit(position.side, last.close, market=True),
                ExitReason.END_OF_DATA,
            )
            trades.append(trade)
            position = None
            # 強制決済をエクイティカーブへ反映する。これを忘れると
            # 最終資産が時価評価のまま残り、実現損益と食い違う。
            if curve:
                curve[-1] = EquityPoint(
                    time=last.time, equity=equity, realized_equity=equity
                )

        return BacktestResult(
            trades=trades,
            equity_curve=curve,
            config=self.config,
            signals_generated=signals_generated,
            signals_taken=signals_taken,
            rejections=strategy.rejection_summary(),
            gate_rejections=gate_rejections,
            halt_reason=risk_manager.halt_reason,
            bars=len(candles),
            start_time=candles[0].time if candles else None,
            end_time=candles[-1].time if candles else None,
        )

    # ------------------------------------------------------------------
    # 執行モデル(ペーパー取引と共有: llmfx/execution/fills.py)
    # ------------------------------------------------------------------
    def _open_position(
        self,
        signal: Signal,
        candle: Candle,
        equity: float,
        fills: FillModel,
        index: int,
        gate_decision: dict | None,
        entry_note: dict | None,
        fill_price_source: str = "open",
    ) -> Position | None:
        base_price = candle.open if fill_price_source == "open" else candle.close
        entry_price = fills.entry(signal.side, base_price)

        # 約定価格がずれた結果、損切りの向きが破綻していないか確認する。
        if signal.side is Side.LONG and entry_price <= signal.stop_loss:
            return None
        if signal.side is Side.SHORT and entry_price >= signal.stop_loss:
            return None

        risk_fraction = min(
            self.config.risk.risk_per_trade, self.config.risk.max_risk_per_trade
        )
        sizing_equity = (
            equity if self.config.risk.compounding else self.config.risk.initial_equity
        )
        units, risk_amount = position_size(
            equity=sizing_equity,
            risk_fraction=risk_fraction,
            entry_price=entry_price,
            stop_price=signal.stop_loss,
            quote_to_account_rate=self.config.instrument.quote_to_account_rate,
        )
        if units <= 0:
            return None

        gate_payload = dict(gate_decision) if gate_decision else None
        return Position(
            signal=signal,
            side=signal.side,
            units=units,
            entry_price=entry_price,
            entry_time=candle.time,
            entry_index=index,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            initial_risk_per_unit=abs(entry_price - signal.stop_loss),
            risk_amount=risk_amount,
            entry_note={"gate": gate_payload, **(entry_note or {})}
            if (gate_payload or entry_note)
            else None,
        )

    def _manage_stop(
        self, position: Position, strategy: DowReversalStrategy, candle: Candle
    ) -> None:
        anchor = strategy.analyzer.detector.last_swing(
            SwingType.LOW if position.side is Side.LONG else SwingType.HIGH
        )
        update_stop(
            position=position,
            candle=candle,
            config=self.config,
            structure_anchor=anchor.price if anchor is not None else None,
            atr=strategy.atr or 0.0,
        )

    def _close_position(
        self,
        position: Position,
        equity: float,
        candle: Candle,
        index: int,
        exit_price: float,
        reason: ExitReason,
    ) -> tuple[Trade, float]:
        rate = self.config.instrument.quote_to_account_rate
        gross = (
            (exit_price - position.entry_price) * position.side.sign * position.units * rate
        )
        commission = self.config.execution.commission_per_unit * position.units
        pnl = gross - commission
        new_equity = equity + pnl
        r_multiple = pnl / position.risk_amount if position.risk_amount > 0 else 0.0

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
            r_multiple=r_multiple,
            exit_reason=reason,
            bars_held=index - position.entry_index,
            equity_after=new_equity,
            rr_at_entry=position.signal.rr,
            target_source=position.signal.target_source,
            structure=position.signal.structure,
            max_favorable_excursion=position.max_favorable_excursion,
            max_adverse_excursion=position.max_adverse_excursion,
            entry_note=position.entry_note,
            gate_decision=(position.entry_note or {}).get("gate") if position.entry_note else None,
        )
        return trade, new_equity

    def _mark_to_market(
        self, position: Position | None, candle: Candle, equity: float
    ) -> float:
        if position is None:
            return equity
        rate = self.config.instrument.quote_to_account_rate
        unrealized = (
            (candle.close - position.entry_price)
            * position.side.sign
            * position.units
            * rate
        )
        return equity + unrealized

    def _context(
        self,
        strategy: DowReversalStrategy,
        candle: Candle,
        index: int,
        equity: float,
    ) -> dict[str, Any]:
        recent = strategy.analyzer.swings[-8:]
        return {
            "instrument": self.config.instrument.symbol,
            "granularity": self.config.instrument.granularity,
            "time": candle.time.isoformat(),
            "bar_index": index,
            "close": candle.close,
            "atr": strategy.atr,
            "trend": strategy.trend.value,
            "equity": equity,
            "recent_swings": [
                {
                    "type": s.type.value,
                    "label": s.label.value,
                    "price": s.price,
                    "time": s.time.isoformat(),
                }
                for s in recent
            ],
        }
