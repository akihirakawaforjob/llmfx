"""設定の読み込みと検証のテスト.

リスク設定のタイポが黙って無視されると、想定より大きなリスクで
動き続けることになる。未知のキーは必ずエラーにする。
"""

from __future__ import annotations

import pytest
import yaml

from llmfx.config import AppConfig, ConfigError


def test_defaults_are_valid():
    config = AppConfig.from_dict({})
    assert config.entry.min_rr == 2.0
    assert config.risk.monthly_target == 1.4
    assert config.instrument.symbol == "USD_JPY"


def test_unknown_section_is_rejected():
    with pytest.raises(ConfigError, match="未知の設定セクション"):
        AppConfig.from_dict({"riskk": {"risk_per_trade": 0.5}})


def test_unknown_key_within_a_section_is_rejected():
    """`risk_pre_trade` のようなタイポを黙って通さない。"""
    with pytest.raises(ConfigError, match="未知のキー"):
        AppConfig.from_dict({"risk": {"risk_pre_trade": 0.02}})


def test_risk_per_trade_cannot_exceed_its_own_cap():
    with pytest.raises(ConfigError, match="max_risk_per_trade"):
        AppConfig.from_dict({"risk": {"risk_per_trade": 0.10, "max_risk_per_trade": 0.05}})


def test_risk_per_trade_must_be_a_fraction():
    with pytest.raises(ConfigError):
        AppConfig.from_dict({"risk": {"risk_per_trade": 0.0}})
    with pytest.raises(ConfigError):
        AppConfig.from_dict({"risk": {"risk_per_trade": 1.5, "max_risk_per_trade": 2.0}})


def test_unknown_target_strategy_is_rejected():
    with pytest.raises(ConfigError, match="未知の利確戦略"):
        AppConfig.from_dict({"entry": {"target_strategies": ["fibonacci"]}})


def test_empty_target_strategies_is_rejected():
    with pytest.raises(ConfigError, match="target_strategies"):
        AppConfig.from_dict({"entry": {"target_strategies": []}})


def test_invalid_entry_mode_is_rejected():
    with pytest.raises(ConfigError, match="entry_mode"):
        AppConfig.from_dict({"execution": {"entry_mode": "limit"}})


def test_invalid_stop_basis_mode_is_rejected():
    with pytest.raises(ConfigError, match="stop_basis_mode"):
        AppConfig.from_dict({"entry": {"stop_basis_mode": "whatever"}})


def test_invalid_llm_effort_is_rejected():
    with pytest.raises(ConfigError, match="effort"):
        AppConfig.from_dict({"llm": {"effort": "turbo"}})


def test_round_trip_through_dict():
    original = AppConfig.from_dict({"entry": {"min_rr": 2.5}, "risk": {"risk_per_trade": 0.01}})
    restored = AppConfig.from_dict(original.to_dict())
    assert restored.entry.min_rr == 2.5
    assert restored.risk.risk_per_trade == 0.01


@pytest.mark.parametrize("name", ["default", "aggressive"])
def test_shipped_configs_load(tmp_path, name):
    """同梱の設定ファイルが実際に読み込めること。"""
    from pathlib import Path

    path = Path(__file__).parent.parent / "configs" / f"{name}.yaml"
    config = AppConfig.load(path)
    assert config.entry.min_rr >= 2.0, "要件どおり RR 下限は 2.0 以上であること"


def test_load_from_yaml_file(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        yaml.safe_dump({"instrument": {"symbol": "EUR_USD", "pip_size": 0.0001}}),
        encoding="utf-8",
    )
    config = AppConfig.load(path)
    assert config.instrument.symbol == "EUR_USD"
    assert config.instrument.pip_size == 0.0001
