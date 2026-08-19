"""押し目待ちの状態機械のテスト.

図で合意した手順:
  上位足が上抜ける → 待機 → 押し目 → 下位足のダウ転換で入る → 損切りは押し安値の下

先読みを避けるため、参照するのは確定した足だけ。上位足の転換は
「その上位足が閉じた下位足」でだけ露出される。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llmfx.config import AppConfig, ConfigError
from llmfx.data.synthetic import generate_synthetic_candles
from llmfx.domain.pullback import PendingSetup, PullbackTracker
from llmfx.domain.strategy import DowReversalStrategy
from llmfx.domain.types import Candle, Side

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def bar(i: int, low: float, high: float, close: float | None = None) -> Candle:
    return Candle(
        time=T0 + timedelta(minutes=15 * i),
        open=(low + high) / 2, high=high, low=low,
        close=close if close is not None else (low + high) / 2, volume=1.0,
    )


# --- 状態機械 -------------------------------------------------------------


def test_arming_records_the_break_level_and_starts_tracking_the_extreme():
    t = PullbackTracker()
    t.arm(Side.LONG, break_level=100.0, candle=bar(0, 99.0, 101.0), index=5)
    assert t.pending is not None
    assert t.pending.break_level == 100.0
    assert t.pending.extreme == 99.0


def test_extreme_follows_the_pullback_low_for_a_long_setup():
    t = PullbackTracker()
    t.arm(Side.LONG, 100.0, bar(0, 99.0, 101.0), 0)
    t.observe(bar(1, 98.0, 100.5), max_bars=20, tolerance=10.0)
    t.observe(bar(2, 98.5, 100.0), max_bars=20, tolerance=10.0)
    assert t.pending.extreme == 98.0, "最安値を追えていない"


def test_extreme_follows_the_rally_high_for_a_short_setup():
    t = PullbackTracker()
    t.arm(Side.SHORT, 100.0, bar(0, 99.0, 101.0), 0)
    t.observe(bar(1, 100.0, 102.0), max_bars=20, tolerance=10.0)
    assert t.pending.extreme == 102.0


def test_setup_times_out_and_is_not_chased():
    t = PullbackTracker()
    t.arm(Side.LONG, 100.0, bar(0, 99.0, 101.0), 0)
    for i in range(1, 4):
        assert t.observe(bar(i, 99.0, 101.0), max_bars=3, tolerance=10.0) is None
    assert t.observe(bar(4, 99.0, 101.0), max_bars=3, tolerance=10.0) == "pullback_timed_out"
    assert t.pending is None


def test_deep_pullback_invalidates_the_setup():
    """上抜け前の水準を割ったらダマシとみなす。"""
    t = PullbackTracker()
    t.arm(Side.LONG, 100.0, bar(0, 99.0, 101.0), 0)
    reason = t.observe(bar(1, 96.0, 99.0, close=96.5), max_bars=20, tolerance=2.0)
    assert reason == "pullback_invalidated"
    assert t.pending is None


def test_shallow_retest_does_not_invalidate():
    """上抜け直後に水準へ戻る動きは正常。ここで切ると機会を潰す。"""
    t = PullbackTracker()
    t.arm(Side.LONG, 100.0, bar(0, 99.0, 101.0), 0)
    assert t.observe(bar(1, 99.0, 100.5, close=99.5), max_bars=20, tolerance=2.0) is None
    assert t.pending is not None


def test_direction_must_match():
    t = PullbackTracker()
    t.arm(Side.LONG, 100.0, bar(0, 99.0, 101.0), 0)
    assert t.matches(Side.LONG)
    assert not t.matches(Side.SHORT)


def test_new_break_replaces_the_old_setup():
    t = PullbackTracker()
    t.arm(Side.LONG, 100.0, bar(0, 99.0, 101.0), 0)
    t.arm(Side.LONG, 105.0, bar(9, 104.0, 106.0), 9)
    assert t.pending.break_level == 105.0
    assert t.pending.bars_waited == 0


# --- 設定 -----------------------------------------------------------------


def test_pullback_mode_requires_a_direction_timeframe():
    config = AppConfig()
    config.entry.mode = "pullback"
    with pytest.raises(ConfigError, match="higher_timeframe"):
        config.validate()


def test_pullback_mode_accepts_a_direction_timeframe():
    config = AppConfig()
    config.entry.mode = "pullback"
    config.entry.higher_timeframe = "H1"
    config.validate()


def test_trigger_does_not_require_a_prior_trend_by_default():
    """要求すると、調整が LH/LL を形成し終えるまで引き金が引けない。

    スイング確定に左右 3 本ずつ要るので、実質 30 本以上の調整でないと
    エントリーできず「1 本でも下げれば押し目」という合意と矛盾する。
    """
    config = AppConfig()
    config.entry.mode = "pullback"
    config.entry.higher_timeframe = "H1"
    config.validate()
    assert DowReversalStrategy(config).analyzer.require_prior_trend is False


# --- 通しで動くこと -------------------------------------------------------


def _run(**entry_over):
    candles = generate_synthetic_candles(count=20000, seed=20260810)
    config = AppConfig()
    config.entry.mode = "pullback"
    config.entry.higher_timeframe = "H1"
    config.entry.min_rr = 1e-6
    config.entry.use_take_profit = False
    for key, value in entry_over.items():
        setattr(config.entry, key, value)
    config.validate()
    strategy = DowReversalStrategy(config)
    return [s for c in candles if (s := strategy.update(c))], strategy


def test_signals_are_produced_and_stop_sits_beyond_the_pullback_extreme():
    signals, _ = _run()
    assert signals, "1 件も出ていない"
    for s in signals:
        if s.side is Side.LONG:
            assert s.stop_loss < s.stop_basis, "損切りが押し安値より内側"
            assert s.stop_loss < s.reference_price
        else:
            assert s.stop_loss > s.stop_basis
            assert s.stop_loss > s.reference_price


def test_waiting_longer_finds_more_setups():
    short, _ = _run(pullback_max_bars=20)
    long, _ = _run(pullback_max_bars=60)
    assert len(long) > len(short)


def test_tight_invalidation_kills_setups():
    loose, _ = _run(pullback_invalidation_atr=3.0)
    tight, _ = _run(pullback_invalidation_atr=0.3)
    assert len(tight) < len(loose)


def test_wider_stop_buffer_increases_risk():
    tight, _ = _run(pullback_stop_buffer_atr=0.1)
    wide, _ = _run(pullback_stop_buffer_atr=1.0)
    by_time = {s.time: s for s in tight}
    pairs = [(by_time[s.time], s) for s in wide if s.time in by_time]
    assert pairs
    assert all(w.risk_per_unit > t.risk_per_unit for t, w in pairs)
