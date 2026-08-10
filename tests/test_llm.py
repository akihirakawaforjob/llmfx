"""LLM 層のテスト.

API キー無しでも実行できる範囲を検証する:
  - 構造化出力スキーマが Claude の要件(全項目必須・追加禁止)を満たすか
  - キーが無いとき、例外で止まらず自動的に無効化されるか
  - API 障害時に fail_open / fail_closed が設定どおり働くか
"""

from __future__ import annotations

import json

import pytest

from llmfx.config import AppConfig
from llmfx.domain.types import Side
from llmfx.llm.client import LLMClient, LLMUnavailable, ResponseCache
from llmfx.llm.gate import EntryGate, signal_payload
from llmfx.llm.journalist import Journalist
from llmfx.llm.prompts import system_prompt
from llmfx.llm.schemas import (
    EntryNote,
    ExitNote,
    GateDecision,
    ReviewReport,
    json_schema_for,
)


SCHEMAS = [GateDecision, EntryNote, ExitNote, ReviewReport]


# ----------------------------------------------------------------------
@pytest.mark.parametrize("model", SCHEMAS)
def test_schema_forbids_extra_properties_everywhere(model):
    """構造化出力は全オブジェクトで additionalProperties: false が必要。"""
    schema = json_schema_for(model)

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert "required" in node
                assert set(node["required"]) == set(node.get("properties", {}))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(schema)


@pytest.mark.parametrize("model", SCHEMAS)
def test_schema_drops_unsupported_keywords(model):
    """数値範囲や文字列長は構造化出力が受け付けないので除去されること。"""
    blob = json.dumps(json_schema_for(model))
    for keyword in ("minimum", "maximum", "minLength", "maxLength", "pattern"):
        assert f'"{keyword}"' not in blob


@pytest.mark.parametrize("model", SCHEMAS)
def test_schema_is_json_serialisable(model):
    json.dumps(json_schema_for(model))


# ----------------------------------------------------------------------
def test_client_is_unavailable_without_an_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config = AppConfig.from_dict({"llm": {"enabled": True}})
    client = LLMClient(config.llm, api_key=None)
    assert not client.available
    assert "ANTHROPIC_API_KEY" in client.unavailable_reason


def test_client_is_unavailable_when_disabled_in_config():
    config = AppConfig.from_dict({"llm": {"enabled": False}})
    client = LLMClient(config.llm, api_key="sk-dummy")
    assert not client.available
    assert "無効化" in client.unavailable_reason


def test_structured_raises_when_unavailable():
    config = AppConfig.from_dict({"llm": {"enabled": False}})
    client = LLMClient(config.llm)
    with pytest.raises(LLMUnavailable):
        client.structured("sys", "user", GateDecision)


# ----------------------------------------------------------------------
class _BrokenClient:
    """常に失敗する LLM クライアントの代役。"""

    available = True
    unavailable_reason = None

    def structured(self, **_kwargs):
        raise LLMUnavailable("テスト用の障害")

    def __getattr__(self, _name):
        raise AttributeError


def _dummy_signal():
    from llmfx.data.synthetic import generate_synthetic_candles
    from llmfx.domain.strategy import DowReversalStrategy

    strategy = DowReversalStrategy(AppConfig.from_dict({"entry": {"min_rr": 1.0}}))
    for candle in generate_synthetic_candles(count=4000, seed=31):
        signal = strategy.update(candle)
        if signal is not None:
            return signal
    raise AssertionError("テスト用のシグナルを生成できませんでした")


class _StubClient:
    available = True
    unavailable_reason = None

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def structured(self, system, user, schema, **_kwargs):
        self.calls += 1
        return schema.model_validate(self.payload)


def test_gate_fails_open_when_the_api_is_down():
    config = AppConfig.from_dict({"llm": {"enabled": True, "fail_open": True}})
    gate = EntryGate(_BrokenClient(), config)
    decision = gate.evaluate(_dummy_signal(), {})
    assert decision["approve"] is True
    assert decision["source"] == "fail_open"
    assert gate.failures == 1


def test_gate_fails_closed_when_configured():
    config = AppConfig.from_dict({"llm": {"enabled": True, "fail_open": False}})
    gate = EntryGate(_BrokenClient(), config)
    decision = gate.evaluate(_dummy_signal(), {})
    assert decision["approve"] is False
    assert decision["source"] == "fail_closed"


def test_gate_veto_is_reported():
    config = AppConfig.from_dict({"llm": {"enabled": True}})
    stub = _StubClient(
        {
            "approve": False,
            "confidence": 0.8,
            "primary_reason": "ヒゲだけのブレイク",
            "risk_flags": ["thin_liquidity"],
        }
    )
    gate = EntryGate(stub, config)
    decision = gate.evaluate(_dummy_signal(), {})
    assert decision["approve"] is False
    assert gate.vetoes == 1
    assert gate.summary()["veto_rate"] == 1.0


def test_journalist_returns_none_when_the_api_is_down():
    config = AppConfig.from_dict({"llm": {"enabled": True}})
    journalist = Journalist(_BrokenClient(), config)
    assert journalist.entry_note(_dummy_signal(), {}) is None
    assert journalist.failures == 1


# ----------------------------------------------------------------------
def test_signal_payload_contains_the_decision_inputs():
    payload = signal_payload(_dummy_signal())
    for key in ("side", "stop_loss", "take_profit", "risk_reward", "stop_basis", "structure"):
        assert key in payload
    assert payload["side"] in {Side.LONG.value, Side.SHORT.value}
    json.dumps(payload)  # プロンプトへ埋め込めること


def test_system_prompt_is_stable_for_caching():
    """プロンプトキャッシュのため、同じ設定なら毎回まったく同じ文字列になること。"""
    config = AppConfig.from_dict({})
    assert system_prompt(config, "gate") == system_prompt(config, "gate")
    assert system_prompt(config, "gate") != system_prompt(config, "journal")


def test_system_prompt_mentions_the_monthly_target():
    config = AppConfig.from_dict({"risk": {"monthly_target": 1.4}})
    assert "1.40 倍" in system_prompt(config, "review")


def test_response_cache_round_trip(tmp_path):
    cache = ResponseCache(tmp_path / "cache.sqlite")
    assert cache.get("k") is None
    cache.put("k", {"approve": True})
    assert cache.get("k") == {"approve": True}
    cache.close()
