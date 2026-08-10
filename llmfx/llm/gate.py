"""エントリー拒否権.

ルールが出したシグナルを LLM が見送れるようにする層。判断の主体は
あくまでルール側で、ここは「通す/見送る」の 1 ビットしか返さない。
"""

from __future__ import annotations

import logging

from ..config import AppConfig
from ..domain.types import Signal
from .client import LLMClient, LLMUnavailable
from .prompts import gate_prompt, system_prompt
from .schemas import GateDecision

logger = logging.getLogger(__name__)


def signal_payload(signal: Signal) -> dict:
    return {
        "side": signal.side.value,
        "time": signal.time.isoformat(),
        "reference_price": round(signal.reference_price, 6),
        "stop_loss": round(signal.stop_loss, 6),
        "take_profit": round(signal.take_profit, 6),
        "risk_per_unit": round(signal.risk_per_unit, 6),
        "reward_per_unit": round(signal.reward_per_unit, 6),
        "risk_reward": round(signal.rr, 3),
        "broken_level": round(signal.broken_level, 6),
        "stop_basis": round(signal.stop_basis, 6),
        "target_source": signal.target_source,
        "rule_reason": signal.reason,
        "structure": {
            "trend": signal.structure.trend.value,
            "last_high": signal.structure.last_high,
            "last_low": signal.structure.last_low,
            "prior_high": signal.structure.prior_high,
            "prior_low": signal.structure.prior_low,
            "last_high_label": signal.structure.last_high_label.value,
            "last_low_label": signal.structure.last_low_label.value,
            "atr": signal.structure.atr,
        },
    }


class EntryGate:
    def __init__(self, client: LLMClient, config: AppConfig) -> None:
        self.client = client
        self.config = config
        self._system = system_prompt(config, "gate")
        self.calls = 0
        self.vetoes = 0
        self.failures = 0

    @property
    def active(self) -> bool:
        return self.config.llm.gate_enabled and self.client.available

    def evaluate(self, signal: Signal, context: dict) -> dict:
        if not self.active:
            return {"approve": True, "source": "disabled"}

        self.calls += 1
        try:
            decision = self.client.structured(
                system=self._system,
                user=gate_prompt(signal_payload(signal), context),
                schema=GateDecision,
            )
        except LLMUnavailable as exc:
            self.failures += 1
            approve = self.config.llm.fail_open
            logger.warning(
                "LLM ゲートが応答しませんでした (%s)。fail_open=%s のため %s します。",
                exc,
                self.config.llm.fail_open,
                "エントリー" if approve else "見送り",
            )
            return {
                "approve": approve,
                "source": "fail_open" if approve else "fail_closed",
                "error": str(exc),
            }

        if not decision.approve:
            self.vetoes += 1
        return {
            "approve": decision.approve,
            "confidence": max(0.0, min(1.0, decision.confidence)),
            "primary_reason": decision.primary_reason,
            "risk_flags": decision.risk_flags,
            "source": "llm",
        }

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "vetoes": self.vetoes,
            "failures": self.failures,
            "veto_rate": self.vetoes / self.calls if self.calls else 0.0,
        }
