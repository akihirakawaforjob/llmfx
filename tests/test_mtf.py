"""上位足フィルタのテスト.

最重要は先読み防止。上位足のバーは、その期間の最後の下位足が閉じてから
初めて確定する。「上位足の最後の 1 本を渡した瞬間にはまだ確定しない」
という形で確認する。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llmfx.config import AppConfig, ConfigError
from llmfx.data.synthetic import generate_synthetic_candles
from llmfx.domain.mtf import (
    HigherTimeframeFilter,
    TimeframeError,
    granularity_minutes,
)
from llmfx.domain.strategy import DowReversalStrategy
from llmfx.domain.types import Candle, Trend

UTC = timezone.utc
START = datetime(2024, 1, 1, tzinfo=UTC)


def m15(index: int, open_: float, high: float, low: float, close: float) -> Candle:
    return Candle(
        time=START + timedelta(minutes=15 * index),
        open=open_, high=high, low=low, close=close, volume=1.0,
    )


def flat(index: int, price: float) -> Candle:
    return m15(index, price, price + 1, price - 1, price)


# --- 時間足の表記 --------------------------------------------------------


def test_granularity_minutes():
    assert granularity_minutes("H4") == 240
    assert granularity_minutes("m15") == 15
    assert granularity_minutes("D1") == 1440


def test_unknown_granularity_is_rejected():
    with pytest.raises(TimeframeError, match="未対応"):
        granularity_minutes("H3")


# --- 集約と先読み --------------------------------------------------------


def test_higher_timeframe_bar_is_only_completed_after_the_next_one_starts():
    """H1 = M15 x 4。4 本目を渡した時点ではまだ確定させない。"""
    htf = HigherTimeframeFilter(minutes=60)
    for i in range(4):
        htf.update(flat(i, 100.0))
    assert htf.completed_bars == 0, "形成中の上位足を確定扱いしている(先読み)"

    htf.update(flat(4, 100.0))  # 次の H1 の 1 本目
    assert htf.completed_bars == 1


def test_bars_are_aggregated_into_the_higher_timeframe_ohlc():
    htf = HigherTimeframeFilter(minutes=60)
    for b in (
        m15(0, 100.0, 105.0, 99.0, 104.0),
        m15(1, 104.0, 108.0, 103.0, 106.0),
        m15(2, 106.0, 107.0, 95.0, 96.0),
        m15(3, 96.0, 101.0, 96.0, 100.0),
    ):
        htf.update(b)
    assert htf._open == 100.0
    assert htf._high == 108.0
    assert htf._low == 95.0
    assert htf._close == 100.0


def test_bar_count_matches_the_aggregation_ratio():
    """M15 を 400 本流すと H4(= 16 本分)は 24 本確定する(最後の 1 本は未確定)。"""
    htf = HigherTimeframeFilter(minutes=240)
    for i in range(400):
        htf.update(flat(i, 100.0))
    assert htf.completed_bars == 400 // 16 - 1


# --- バイアスと極値 ------------------------------------------------------


def test_bias_appears_on_real_looking_data():
    candles = generate_synthetic_candles(count=8000, seed=20260810)
    htf = HigherTimeframeFilter(minutes=240)
    for c in candles:
        htf.update(c)

    assert htf.bias in (Trend.UP, Trend.DOWN), "上位足の転換が 1 度も出ていない"
    assert htf.extreme is not None
    assert htf.completed_bars > 100


def test_extreme_moves_only_in_the_direction_the_bias_implies():
    """上昇バイアスなら最安値の追従(下がる方向にしか動かない)、下降なら最高値。"""
    candles = generate_synthetic_candles(count=8000, seed=20260810)
    htf = HigherTimeframeFilter(minutes=240)

    previous_bias = None
    previous_extreme = None
    for c in candles:
        htf.update(c)
        if htf.bias is None:
            continue
        if htf.bias is previous_bias and previous_extreme is not None:
            if htf.bias is Trend.UP:
                assert htf.extreme <= previous_extreme + 1e-9, "上昇バイアスで安値が上がっている"
            else:
                assert htf.extreme >= previous_extreme - 1e-9, "下降バイアスで高値が下がっている"
        previous_bias, previous_extreme = htf.bias, htf.extreme


def test_bars_since_reversal_resets_on_a_new_reversal():
    candles = generate_synthetic_candles(count=8000, seed=20260810)
    htf = HigherTimeframeFilter(minutes=240)

    seen_reset = False
    previous = 0
    for c in candles:
        htf.update(c)
        if htf.bias is None:
            continue
        if htf.bars_since_reversal < previous:
            seen_reset = True
        previous = htf.bars_since_reversal
    assert seen_reset, "転換が 1 度しか起きておらず、リセットを確認できない"


# --- 設定の検証 ----------------------------------------------------------


def test_higher_timeframe_must_be_above_the_trading_timeframe():
    config = AppConfig()
    config.instrument.granularity = "H4"
    config.entry.higher_timeframe = "M15"
    with pytest.raises(ConfigError, match="上位の足"):
        config.validate()


def test_same_timeframe_is_rejected():
    config = AppConfig()
    config.instrument.granularity = "M15"
    config.entry.higher_timeframe = "M15"
    with pytest.raises(ConfigError, match="上位の足"):
        config.validate()


def test_strategy_builds_no_filter_when_higher_timeframe_is_none():
    assert DowReversalStrategy(AppConfig()).htf is None


def test_strategy_builds_the_filter_when_configured():
    config = AppConfig()
    config.entry.higher_timeframe = "H4"
    config.validate()
    strategy = DowReversalStrategy(config)
    assert strategy.htf is not None
    assert strategy.htf.seconds == 240 * 60


# --- フィルタが効いていること ---------------------------------------------


def test_alignment_filter_rejects_counter_trend_signals():
    candles = generate_synthetic_candles(count=8000, seed=20260810)

    base = AppConfig()
    base.entry.min_rr = 1e-6
    plain = DowReversalStrategy(base)
    all_signals = [s for c in candles if (s := plain.update(c))]

    filtered_cfg = AppConfig.from_dict(base.to_dict())
    filtered_cfg.entry.higher_timeframe = "H4"
    filtered_cfg.entry.require_htf_alignment = True
    filtered_cfg.validate()
    filtered = DowReversalStrategy(filtered_cfg)
    kept = [s for c in candles if (s := filtered.update(c))]

    reasons = {r.reason for r in filtered.rejections}
    assert "htf_not_aligned" in reasons or "htf_no_bias" in reasons
    assert len(kept) < len(all_signals), "フィルタが 1 件も落としていない"


def test_proximity_filter_rejects_entries_far_from_the_extreme():
    candles = generate_synthetic_candles(count=8000, seed=20260810)

    loose = AppConfig()
    loose.entry.min_rr = 1e-6
    loose.entry.higher_timeframe = "H4"
    loose.validate()
    loose_kept = [s for c in candles if (s := DowReversalStrategy(loose).update(c))]

    tight_cfg = AppConfig.from_dict(loose.to_dict())
    tight_cfg.entry.htf_proximity_atr = 0.1
    tight_cfg.validate()
    tight = DowReversalStrategy(tight_cfg)
    tight_kept = [s for c in candles if (s := tight.update(c))]

    assert len(tight_kept) <= len(loose_kept)
    assert any(r.reason == "htf_too_far_from_extreme" for r in tight.rejections)
