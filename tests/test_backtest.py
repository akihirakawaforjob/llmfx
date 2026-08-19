"""バックテストエンジンのテスト.

最重要は先読み(look-ahead bias)が無いこと。将来の足を覗いていれば、
データを途中で打ち切ったときに過去のトレードまで変わってしまう。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llmfx.backtest.engine import BacktestEngine
from llmfx.backtest.metrics import compute_stats
from llmfx.config import AppConfig
from llmfx.data.synthetic import generate_synthetic_candles
from llmfx.domain.types import Candle, ExitReason, Position, Side, Signal, StructureSnapshot, SwingLabel, Trend
from llmfx.execution.fills import (
    FillModel,
    evaluate_exit,
    rollovers_crossed,
    trade_costs,
)

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ----------------------------------------------------------------------
def test_no_lookahead_truncating_data_does_not_change_past_trades():
    """データを途中で切っても、それ以前のトレードは 1 件も変わらないこと。"""
    config = AppConfig.from_dict({})
    candles = generate_synthetic_candles(count=8000, seed=11)

    full = BacktestEngine(config).run(candles)
    truncated = BacktestEngine(config).run(candles[:5000])

    cutoff = candles[4999].time
    # 打ち切り点より前に決済が完了したトレードだけを比較する。
    full_past = [t for t in full.trades if t.exit_time < cutoff]
    trunc_past = [t for t in truncated.trades if t.exit_time < cutoff]

    assert full_past, "比較できるだけのトレードが発生していること"
    assert len(full_past) == len(trunc_past)
    for a, b in zip(full_past, trunc_past):
        assert a.entry_time == b.entry_time
        assert a.entry_price == pytest.approx(b.entry_price)
        assert a.exit_time == b.exit_time
        assert a.exit_price == pytest.approx(b.exit_price)
        assert a.pnl == pytest.approx(b.pnl)


def test_backtest_is_deterministic():
    config = AppConfig.from_dict({})
    candles = generate_synthetic_candles(count=4000, seed=3)
    first = BacktestEngine(config).run(candles)
    second = BacktestEngine(config).run(candles)
    assert [t.pnl for t in first.trades] == [t.pnl for t in second.trades]


def test_wider_spread_reduces_returns():
    candles = generate_synthetic_candles(count=8000, seed=5)
    cheap = BacktestEngine(AppConfig.from_dict({"execution": {"spread_pips": 0.5}})).run(candles)
    costly = BacktestEngine(AppConfig.from_dict({"execution": {"spread_pips": 6.0}})).run(candles)
    assert compute_stats(costly).total_return < compute_stats(cheap).total_return


def test_equity_curve_covers_every_bar():
    candles = generate_synthetic_candles(count=2000, seed=9)
    result = BacktestEngine(AppConfig.from_dict({})).run(candles)
    assert len(result.equity_curve) == len(candles)
    assert result.equity_curve[0].time == candles[0].time
    assert result.equity_curve[-1].time == candles[-1].time


def test_no_position_survives_the_end_of_data():
    candles = generate_synthetic_candles(count=3000, seed=13)
    result = BacktestEngine(AppConfig.from_dict({})).run(candles)
    # 全建玉が決済されていれば、最終的な実現損益と時価評価が一致する。
    assert result.equity_curve[-1].equity == pytest.approx(
        result.equity_curve[-1].realized_equity
    )


def test_realized_pnl_matches_final_equity():
    config = AppConfig.from_dict({})
    result = BacktestEngine(config).run(generate_synthetic_candles(count=5000, seed=17))
    expected = config.risk.initial_equity + sum(t.pnl for t in result.trades)
    assert result.equity_curve[-1].realized_equity == pytest.approx(expected)


def test_drawdown_stop_halts_trading():
    config = AppConfig.from_dict(
        {"risk": {"max_drawdown_stop": 0.001, "risk_per_trade": 0.05}}
    )
    result = BacktestEngine(config).run(generate_synthetic_candles(count=6000, seed=23))
    if result.trades:
        assert result.halt_reason is not None


# ----------------------------------------------------------------------
def _position(side: Side, entry: float, stop: float, target: float) -> Position:
    structure = StructureSnapshot(
        trend=Trend.DOWN,
        last_high=entry,
        last_low=stop,
        prior_high=None,
        prior_low=None,
        last_high_label=SwingLabel.LH,
        last_low_label=SwingLabel.LL,
        atr=0.5,
        swing_count=4,
    )
    signal = Signal(
        time=START,
        bar_index=0,
        side=side,
        reference_price=entry,
        stop_loss=stop,
        take_profit=target,
        risk_per_unit=abs(entry - stop),
        reward_per_unit=abs(target - entry),
        rr=abs(target - entry) / abs(entry - stop),
        broken_level=entry,
        stop_basis=stop,
        target_source="test",
        structure=structure,
        reason="test",
    )
    return Position(
        signal=signal,
        side=side,
        units=1000,
        entry_price=entry,
        entry_time=START,
        entry_index=0,
        stop_loss=stop,
        take_profit=target,
        initial_risk_per_unit=abs(entry - stop),
        risk_amount=100.0,
    )


def test_stop_wins_when_both_stop_and_target_are_hit_in_one_bar():
    """同一足で両方に触れたら、必ず損切りが先に約定したものとして扱う。"""
    position = _position(Side.LONG, entry=100.0, stop=99.0, target=102.0)
    bar = Candle(time=START + timedelta(minutes=15), open=100.0, high=102.5, low=98.5, close=101.0)
    fills = FillModel(pip_size=0.01, spread_pips=0.0, slippage_pips=0.0)

    result = evaluate_exit(position, bar, fills, bar_index=1, max_bars_in_trade=0)
    assert result is not None
    price, reason = result
    assert reason is ExitReason.STOP_LOSS
    assert price == pytest.approx(99.0)


def test_gap_through_the_stop_fills_at_the_open():
    position = _position(Side.LONG, entry=100.0, stop=99.0, target=102.0)
    bar = Candle(time=START + timedelta(minutes=15), open=97.0, high=97.5, low=96.0, close=96.5)
    fills = FillModel(pip_size=0.01, spread_pips=0.0, slippage_pips=0.0)

    result = evaluate_exit(position, bar, fills, bar_index=1, max_bars_in_trade=0)
    assert result is not None
    price, reason = result
    assert reason is ExitReason.STOP_LOSS
    assert price == pytest.approx(97.0), "窓を開けたら損切り水準ではなく始値で約定する"


def test_slippage_applies_to_stops_but_not_to_targets():
    fills = FillModel(pip_size=0.01, spread_pips=2.0, slippage_pips=1.0)
    cost = fills.cost_per_side(99.0)  # (2/2 + 1) * 0.01 = 0.02

    stop_fill = fills.exit(Side.LONG, 99.0, market=True)
    target_fill = fills.exit(Side.LONG, 102.0, market=False)
    assert stop_fill == pytest.approx(99.0 - cost)
    assert target_fill == pytest.approx(102.0)


def test_entry_fill_is_adverse_for_both_sides():
    fills = FillModel(pip_size=0.01, spread_pips=2.0, slippage_pips=0.0)
    assert fills.entry(Side.LONG, 100.0) > 100.0
    assert fills.entry(Side.SHORT, 100.0) < 100.0


# --- 価格比例コストと建玉管理料 -------------------------------------------
# 暗号資産は価格水準が数倍動くため固定 pips ではコストを表せず、
# レバレッジ手数料(0.04%/日)は数日保有すると効いてくる。


def test_proportional_spread_scales_with_the_price_level():
    fills = FillModel(pip_size=1.0, spread_pips=0.0, slippage_pips=0.0, spread_bps=2.0)
    assert fills.cost_per_side(10_000_000.0) == pytest.approx(1_000.0)
    assert fills.cost_per_side(3_000_000.0) == pytest.approx(300.0)


def test_fixed_and_proportional_costs_add_up():
    fills = FillModel(pip_size=1.0, spread_pips=100.0, slippage_pips=50.0, spread_bps=2.0)
    # 固定 (100/2 + 50) * 1.0 = 100、比例 10,000,000 * 1bp = 1,000
    assert fills.cost_per_side(10_000_000.0) == pytest.approx(1_100.0)


def test_holding_cost_counts_only_the_rollovers_actually_crossed():
    """GMO は日本時間 6:00(UTC 21:00)をまたぐたびに課金する。"""
    day = datetime(2024, 3, 10, tzinfo=timezone.utc)
    assert rollovers_crossed(day.replace(hour=10), day.replace(hour=20), 21) == 0
    assert rollovers_crossed(day.replace(hour=10), day.replace(hour=23), 21) == 1
    assert rollovers_crossed(
        day.replace(hour=22), day.replace(hour=20) + timedelta(days=1), 21
    ) == 0
    assert rollovers_crossed(
        day.replace(hour=10), day.replace(hour=22) + timedelta(days=2), 21
    ) == 3


def test_holding_cost_is_charged_on_notional_per_night():
    config = AppConfig()
    config.execution.daily_holding_cost_bps = 4.0  # GMO 暗号資産FX 0.04%/日
    day = datetime(2024, 3, 10, tzinfo=timezone.utc)
    costs = trade_costs(
        config, units=0.1, entry_price=10_000_000.0, exit_price=10_000_000.0,
        entry_time=day.replace(hour=10), exit_time=day.replace(hour=23),
    )
    # 建玉評価額 1,000,000 円 x 4bp x 1 泊 = 400 円
    assert costs.commission == pytest.approx(0.0)
    assert costs.holding == pytest.approx(400.0)


def test_no_holding_cost_when_the_rate_is_zero():
    """FX のスワップは方向で符号が変わるため、この一律コストは既定で無効。"""
    config = AppConfig()
    day = datetime(2024, 3, 10, tzinfo=timezone.utc)
    costs = trade_costs(
        config, units=1000.0, entry_price=150.0, exit_price=151.0,
        entry_time=day.replace(hour=10), exit_time=day + timedelta(days=5),
    )
    assert costs.holding == pytest.approx(0.0)


def test_commission_bps_is_charged_on_both_legs():
    """GMO 取引所現物の Taker 0.05% は往復で 0.1% になる。"""
    config = AppConfig()
    config.execution.commission_bps = 5.0
    day = datetime(2024, 3, 10, tzinfo=timezone.utc)
    costs = trade_costs(
        config, units=0.1, entry_price=10_000_000.0, exit_price=10_000_000.0,
        entry_time=day.replace(hour=10), exit_time=day.replace(hour=12),
    )
    assert costs.commission == pytest.approx(1_000.0)
