"""ストラテジーのフィルタが要件どおり効いているかのテスト."""

from __future__ import annotations

import numpy as np
import pytest

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


# --- 売買方向の制限 -------------------------------------------------------
# 暗号資産の現物は売り建てができない。レバレッジと比較するために必要。


def test_disallowing_shorts_drops_only_short_signals():
    candles = generate_synthetic_candles(count=6000, seed=20260810)

    both = AppConfig()
    both.entry.min_rr = 1e-6
    plain = DowReversalStrategy(both)
    all_signals = [s for c in candles if (s := plain.update(c))]

    long_only = AppConfig.from_dict(both.to_dict())
    long_only.entry.allow_short = False
    strategy = DowReversalStrategy(long_only)
    kept = [s for c in candles if (s := strategy.update(c))]

    assert all(s.side is Side.LONG for s in kept)
    assert len(kept) == sum(1 for s in all_signals if s.side is Side.LONG)
    assert any(r.reason == "short_not_allowed" for r in strategy.rejections)


def test_both_directions_disabled_is_a_config_error():
    from llmfx.config import ConfigError

    config = AppConfig()
    config.entry.allow_long = False
    config.entry.allow_short = False
    with pytest.raises(ConfigError, match="allow_long"):
        config.validate()


# --- 利確を置かない設定 ---------------------------------------------------
# 固定の利確は勝ちの頭を押さえる。実測(FX 3 銘柄 20 年)では、利確を外すと
# 平均勝ちが 4.15 R から 7.96 R へ、最大勝ちが 20 R から 52 R へ伸びた。


def test_take_profit_is_used_by_default():
    assert AppConfig().entry.use_take_profit is True


def test_disabling_take_profit_pushes_the_target_out_of_reach():
    """約定判定のコードは触らず、届かない水準へ逃がして無効化する。"""
    candles = generate_synthetic_candles(count=6000, seed=20260810)

    with_tp = AppConfig()
    with_tp.entry.min_rr = 1e-6
    a = DowReversalStrategy(with_tp)
    capped = [s for c in candles if (s := a.update(c))]

    without = AppConfig.from_dict(with_tp.to_dict())
    without.entry.use_take_profit = False
    b = DowReversalStrategy(without)
    running = [s for c in candles if (s := b.update(c))]

    assert len(capped) == len(running), "シグナルの本数は変わらないはず"
    by_time = {s.time: s for s in capped}
    for s in running:
        other = by_time[s.time]
        # 損切りとエントリーは同じ。利確だけが遠のく。
        assert s.stop_loss == pytest.approx(other.stop_loss)
        assert s.reference_price == pytest.approx(other.reference_price)
        if s.side is Side.LONG:
            assert s.take_profit > other.take_profit
        else:
            assert s.take_profit < other.take_profit
        assert s.target_source == "trail_only"


def test_rr_filter_still_applies_without_a_take_profit():
    """伸ばし切る前提でも、行き先の無い場面は見送りたいので選別は残す。"""
    candles = generate_synthetic_candles(count=6000, seed=20260810)
    config = AppConfig()
    config.entry.use_take_profit = False
    config.entry.min_rr = 100.0
    strategy = DowReversalStrategy(config)
    kept = [s for c in candles if (s := strategy.update(c))]
    assert not kept
    assert any(r.reason == "rr_below_minimum" for r in strategy.rejections)


# --- レンジを触らない -----------------------------------------------------


def test_range_filter_only_removes_signals():
    """レンジ除外は選別であって、新しいシグナルを作ってはいけない。"""
    _s, loose = collect_signals({"entry": {"skip_range_structure": False}})
    _s, strict = collect_signals({"entry": {"skip_range_structure": True}})
    assert len(strict) <= len(loose)
    kept = {(s.time, s.side) for s in strict}
    assert kept <= {(s.time, s.side) for s in loose}, "元に無いシグナルが生えている"


def test_range_filter_leaves_no_range_entries_behind():
    """通したシグナルの時点で、構造がレンジであってはならない。"""
    from llmfx.domain.types import Trend

    config = AppConfig.from_dict({"entry": {"skip_range_structure": True}})
    strategy = DowReversalStrategy(config)
    checked = 0
    for candle in generate_synthetic_candles(count=6000, seed=7):
        if strategy.update(candle) is not None:
            assert strategy.analyzer.structure_trend() is not Trend.RANGE
            checked += 1
    assert checked > 0, "検証できるだけのシグナルが出ていること"


def test_range_filter_records_why_it_declined():
    strategy, _signals = collect_signals({"entry": {"skip_range_structure": True}})
    reasons = {r.reason for r in strategy.rejections}
    assert "structure_is_range" in reasons
