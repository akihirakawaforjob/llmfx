"""ペーパー / ライブ実行ループ.

バックテストと同じ戦略オブジェクト・同じ約定モデルを使い、足の供給元
だけを差し替える。これにより「バックテストでは動いたのに実運用で挙動が
違う」という事故の入り込む余地を減らす。

足の供給元(CandleFeed)は 2 種類:
  - ReplayFeed : CSV を時系列に再生する。API 不要でループ全体を検証できる
  - OandaFeed  : OANDA から確定足をポーリングする
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Protocol

from ..config import AppConfig
from ..domain.risk import RiskManager
from ..domain.strategy import DowReversalStrategy
from ..domain.types import Candle, ExitReason, SwingType, Trade
from .broker import Broker
from .fills import update_stop

logger = logging.getLogger(__name__)

GRANULARITY_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D": 86400,
}


class CandleFeed(Protocol):
    def warmup(self, count: int) -> list[Candle]:
        """戦略を暖機するための過去足(確定足のみ)。"""

    def poll(self) -> list[Candle]:
        """前回以降に確定した足。無ければ空リスト。"""

    def finished(self) -> bool:
        """再生型フィードが終端に達したか。ライブでは常に False。"""


class ReplayFeed:
    """既存の足を時系列順に 1 本ずつ供給する。オフライン検証用。"""

    def __init__(self, candles: list[Candle], warmup_bars: int = 200) -> None:
        self._candles = candles
        self._cursor = min(warmup_bars, len(candles))
        self._warmup_bars = self._cursor

    def warmup(self, count: int) -> list[Candle]:
        return self._candles[: self._warmup_bars]

    def poll(self) -> list[Candle]:
        if self._cursor >= len(self._candles):
            return []
        candle = self._candles[self._cursor]
        self._cursor += 1
        return [candle]

    def finished(self) -> bool:
        return self._cursor >= len(self._candles)


class OandaFeed:
    """OANDA から確定足を取得する。未確定足は client 側で除外済み。"""

    def __init__(self, client, instrument: str, granularity: str) -> None:
        self.client = client
        self.instrument = instrument
        self.granularity = granularity
        self._last_time: datetime | None = None

    def warmup(self, count: int) -> list[Candle]:
        candles = self.client.latest_candles(self.instrument, self.granularity, count=count)
        if candles:
            self._last_time = candles[-1].time
        return candles

    def poll(self) -> list[Candle]:
        candles = self.client.latest_candles(self.instrument, self.granularity, count=10)
        if not candles:
            return []
        if self._last_time is None:
            self._last_time = candles[-1].time
            return []
        fresh = [c for c in candles if c.time > self._last_time]
        if fresh:
            self._last_time = fresh[-1].time
        return fresh

    def finished(self) -> bool:
        return False


@dataclass
class RunnerStats:
    bars_processed: int = 0
    signals: int = 0
    entries: int = 0
    exits: int = 0
    gate_vetoes: int = 0
    blocked_by_risk: int = 0
    trades: list[Trade] = field(default_factory=list)


class TradingRunner:
    """1 銘柄・1 建玉の実行ループ。"""

    def __init__(
        self,
        config: AppConfig,
        feed: CandleFeed,
        broker: Broker,
        gate=None,
        journalist=None,
        on_trade: Callable[[Trade], None] | None = None,
        on_entry: Callable[[object], None] | None = None,
    ) -> None:
        self.config = config
        self.feed = feed
        self.broker = broker
        self.gate = gate
        self.journalist = journalist
        self.on_trade = on_trade
        self.on_entry = on_entry

        self.strategy = DowReversalStrategy(config)
        self.risk_manager = RiskManager(
            initial_equity=broker.equity(),
            max_daily_loss=config.risk.max_daily_loss,
            max_drawdown_stop=config.risk.max_drawdown_stop,
        )
        self.stats = RunnerStats()
        self._bar_index = -1

    # ------------------------------------------------------------------
    def warmup(self, count: int = 500) -> int:
        """過去足を戦略へ流し込む。この間はエントリーしない。"""
        candles = self.feed.warmup(count)
        for candle in candles:
            self._bar_index += 1
            self.strategy.update(candle)
        logger.info("暖機完了: %s 本、現在のトレンド=%s", len(candles), self.strategy.trend.value)
        return len(candles)

    def run(
        self,
        max_bars: int | None = None,
        poll_interval: float | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> RunnerStats:
        interval = poll_interval if poll_interval is not None else self._default_interval()
        processed = 0

        while True:
            if max_bars is not None and processed >= max_bars:
                break
            if self.feed.finished():
                break

            candles = self.feed.poll()
            if not candles:
                if self.feed.finished():
                    break
                sleeper(interval)
                continue

            for candle in candles:
                self.process_candle(candle)
                processed += 1
                if max_bars is not None and processed >= max_bars:
                    break

        return self.stats

    # ------------------------------------------------------------------
    def process_candle(self, candle: Candle) -> None:
        self._bar_index += 1
        self.stats.bars_processed += 1
        index = self._bar_index

        # 1) 建玉の決済判定(ブローカー側で約定済みかどうかを含む)。
        trade = self.broker.on_candle(candle, index)
        if trade is not None:
            self._handle_exit(trade, candle, index)

        # 2) 戦略更新。
        signal = self.strategy.update(candle)

        # 3) 損切りの追従。
        position = self.broker.position()
        if position is not None:
            anchor = self.strategy.analyzer.detector.last_swing(
                SwingType.LOW if position.side.value == "long" else SwingType.HIGH
            )
            update_stop(
                position=position,
                candle=candle,
                config=self.config,
                structure_anchor=anchor.price if anchor is not None else None,
                atr=self.strategy.atr or 0.0,
            )

        # 4) 逆方向のダウ転換で手仕舞う。
        flip = self.strategy.last_event
        position = self.broker.position()
        if (
            position is not None
            and self.config.execution.exit_on_structure_flip
            and flip is not None
            and flip.bar_index == index
            and flip.side is not position.side
        ):
            trade = self.broker.close_position(candle, index, ExitReason.STRUCTURE_FLIP)
            if trade is not None:
                self._handle_exit(trade, candle, index)

        # 5) リスク制限。
        equity = self.broker.equity()
        self.risk_manager.on_bar(equity, candle.time.date())
        if self.risk_manager.halted:
            position = self.broker.position()
            if position is not None:
                trade = self.broker.close_position(candle, index, ExitReason.RISK_KILL_SWITCH)
                if trade is not None:
                    self._handle_exit(trade, candle, index)
            return

        # 6) 新規エントリー。
        if signal is None:
            return
        self.stats.signals += 1
        if self.broker.position() is not None:
            return

        allowed, reason = self.risk_manager.can_open(equity)
        if not allowed:
            self.stats.blocked_by_risk += 1
            logger.info("リスク制限によりエントリーを見送りました: %s", reason)
            return

        context = self._context(candle, index, equity)
        gate_decision = None
        if self.gate is not None:
            gate_decision = self.gate.evaluate(signal, context)
            if gate_decision is not None and not gate_decision.get("approve", True):
                self.stats.gate_vetoes += 1
                logger.info(
                    "LLM がエントリーを見送りました: %s",
                    gate_decision.get("primary_reason", ""),
                )
                return

        entry_note = None
        if self.journalist is not None:
            entry_note = self.journalist.entry_note(signal, context)

        position = self.broker.submit(signal, candle, index)
        if position is None:
            return
        if gate_decision or entry_note:
            position.entry_note = {"gate": gate_decision, **(entry_note or {})}
        self.stats.entries += 1
        logger.info(
            "エントリー: %s %s units @ %.5f SL=%.5f TP=%.5f RR=%.2f",
            position.side.value,
            position.units,
            position.entry_price,
            position.stop_loss,
            position.take_profit,
            signal.rr,
        )
        if self.on_entry is not None:
            self.on_entry(position)

    # ------------------------------------------------------------------
    def _handle_exit(self, trade: Trade, candle: Candle, index: int) -> None:
        self.stats.exits += 1
        self.stats.trades.append(trade)
        if self.journalist is not None:
            trade.exit_note = self.journalist.exit_note(
                trade, self._context(candle, index, self.broker.equity())
            )
        logger.info(
            "決済: %s pnl=%.2f (%.2fR) 理由=%s",
            trade.side.value,
            trade.pnl,
            trade.r_multiple,
            trade.exit_reason.value,
        )
        if self.on_trade is not None:
            self.on_trade(trade)

    def _context(self, candle: Candle, index: int, equity: float) -> dict:
        recent = self.strategy.analyzer.swings[-8:]
        return {
            "instrument": self.config.instrument.symbol,
            "granularity": self.config.instrument.granularity,
            "time": candle.time.isoformat(),
            "bar_index": index,
            "close": candle.close,
            "atr": self.strategy.atr,
            "trend": self.strategy.trend.value,
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

    def _default_interval(self) -> float:
        seconds = GRANULARITY_SECONDS.get(self.config.instrument.granularity.upper(), 900)
        # 足の確定を跨いで取りこぼさない程度の間隔。
        return max(5.0, seconds / 15.0)
