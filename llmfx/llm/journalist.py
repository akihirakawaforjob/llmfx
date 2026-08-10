"""トレード所感の記録.

要件『エントリー時および決済時の所感(なぜ買ったか、なぜ負けたかの分析)
を書いて提出させ、それを改善に使う』の実装。ここで書かれた内容は
journal ストアに保存され、`llmfx journal review` で改善レポートの
入力になる。
"""

from __future__ import annotations

import logging

from ..config import AppConfig
from ..domain.types import Signal, Trade
from .client import LLMClient, LLMUnavailable
from .gate import signal_payload
from .prompts import entry_note_prompt, exit_note_prompt, system_prompt
from .schemas import EntryNote, ExitNote

logger = logging.getLogger(__name__)


def trade_payload(trade: Trade) -> dict:
    return {
        "side": trade.side.value,
        "entry_time": trade.entry_time.isoformat(),
        "exit_time": trade.exit_time.isoformat(),
        "entry_price": round(trade.entry_price, 6),
        "exit_price": round(trade.exit_price, 6),
        "stop_loss": round(trade.stop_loss, 6),
        "take_profit": round(trade.take_profit, 6),
        "risk_reward_at_entry": round(trade.rr_at_entry, 3),
        "target_source": trade.target_source,
        "pnl": round(trade.pnl, 2),
        "r_multiple": round(trade.r_multiple, 3),
        "exit_reason": trade.exit_reason.value,
        "bars_held": trade.bars_held,
        "max_favorable_excursion": round(trade.max_favorable_excursion, 6),
        "max_adverse_excursion": round(trade.max_adverse_excursion, 6),
        "entry_note": trade.entry_note,
        "structure_at_entry": (
            {
                "trend": trade.structure.trend.value,
                "last_high": trade.structure.last_high,
                "last_low": trade.structure.last_low,
                "last_high_label": trade.structure.last_high_label.value,
                "last_low_label": trade.structure.last_low_label.value,
                "atr": trade.structure.atr,
            }
            if trade.structure
            else None
        ),
    }


class Journalist:
    def __init__(self, client: LLMClient, config: AppConfig) -> None:
        self.client = client
        self.config = config
        self._system = system_prompt(config, "journal")
        self.entry_notes = 0
        self.exit_notes = 0
        self.failures = 0

    @property
    def active(self) -> bool:
        return self.config.llm.journal_enabled and self.client.available

    def entry_note(self, signal: Signal, context: dict) -> dict | None:
        if not self.active:
            return None
        try:
            note = self.client.structured(
                system=self._system,
                user=entry_note_prompt(signal_payload(signal), context),
                schema=EntryNote,
            )
        except LLMUnavailable as exc:
            self.failures += 1
            logger.warning("エントリー所感の生成に失敗しました: %s", exc)
            return None
        self.entry_notes += 1
        return note.model_dump()

    def exit_note(self, trade: Trade, context: dict) -> dict | None:
        if not self.active:
            return None
        try:
            note = self.client.structured(
                system=self._system,
                user=exit_note_prompt(trade_payload(trade), context),
                schema=ExitNote,
            )
        except LLMUnavailable as exc:
            self.failures += 1
            logger.warning("決済所感の生成に失敗しました: %s", exc)
            return None
        self.exit_notes += 1
        return note.model_dump()
