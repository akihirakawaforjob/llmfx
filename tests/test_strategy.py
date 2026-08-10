"""ストラテジーのフィルタが要件どおり効いているかのテスト."""

from __future__ import annotations

import numpy as np

from llmfx.config import AppConfig
from llmfx.data.synthetic import generate_synthetic_candles
from llmfx.domain.strategy import DowReversalStrategy
from llmfx.domain.types import Side


def run_strategy(overrides: dict) -> DowReversalStrategy:
    strategy = DowReversalStrategy(AppConfig.from_dict(overrides))
    for candle in generate_synthetic_candles(count=6000, seed=7):
        strategy.update(candle)
    return strategy


def collect_signals(overrides: dict):
    strategy = DowReversalStrategy(AppConfig.from_dict(overrides))
    signals = []
    for candle in generate_synthetic_candles(count=6000, seed=7):
        signal = strategy.update(candle)
        if signal is not None:
            signals.append(signal)
    return strategy, signals


def test_every_signal_meets_the_minimum_risk_reward():
    """要件 3: RR が下限未満のシグナルは 1 つも通してはいけない。"""
    _strategy, signals = collect_signals({"entry": {"min_rr": 2.0}})
    assert signals, "検証できるだけのシグナルが出ていること"
    for signal in signals:
        assert signal.rr >= 2.0, f"RR={signal.rr} が下限 2.0 を下回っている"


def test_raising_min_rr_never_increases_signal_count():
    counts = []
    for threshold in (1.0, 2.0, 3.0, 4.0):
        _strategy, signals = collect_signals({"entry": {"min_rr": threshold}})
        counts.append(len(signals))
    assert counts == sorted(counts, reverse=True), (
        f"min_rr を上げたのにシグナルが増えている: {counts}"
    )


def test_stop_is_always_on_the_losing_side_of_entry():
    """要件 2: 損切りは必ず建値の逆側にあり、リスクは正の値になる。"""
    _strategy, signals = collect_signals({"entry": {"min_rr": 1.0}})
    for signal in signals:
        if signal.side is Side.LONG:
            assert signal.stop_loss < signal.reference_price
            assert signal.take_profit > signal.reference_price
            assert signal.stop_basis >= signal.stop_loss  # バッファの分だけ外側
        else:
            assert signal.stop_loss > signal.reference_price
            assert signal.take_profit < signal.reference_price
            assert signal.stop_basis <= signal.stop_loss
        assert signal.risk_per_unit > 0
        assert abs(signal.rr - signal.reward_per_unit / signal.risk_per_unit) < 1e-9


def test_reported_rr_matches_the_actual_price_levels():
    _strategy, signals = collect_signals({"entry": {"min_rr": 1.5}})
    for signal in signals:
        risk = abs(signal.reference_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.reference_price)
        assert abs(risk - signal.risk_per_unit) < 1e-9
        assert abs(reward - signal.reward_per_unit) < 1e-9


def test_break_extension_filter_rejects_late_entries():
    """ブレイク水準から離れすぎた『飛び乗り』は却下される。"""
    strict = run_strategy({"entry": {"max_break_extension_atr": 0.1}})
    loose = run_strategy({"entry": {"max_break_extension_atr": 10.0}})
    strict_rejects = strict.rejection_summary().get("break_extension_too_large", 0)
    loose_rejects = loose.rejection_summary().get("break_extension_too_large", 0)
    assert strict_rejects > loose_rejects


def test_trend_origin_targets_are_further_than_nearest_structure():
    """トレンド起点の目標は、最も近い壁より必ず遠い(同値以上)。"""
    _s1, nearest = collect_signals(
        {"entry": {"min_rr": 0.01, "target_strategies": ["structure"]}}
    )
    _s2, origin = collect_signals(
        {"entry": {"min_rr": 0.01, "target_strategies": ["trend_origin"]}}
    )
    assert nearest and origin
    assert np.median([s.rr for s in origin]) > np.median([s.rr for s in nearest])
