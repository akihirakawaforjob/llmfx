"""リスク計算と目標月利の逆算のテスト.

ここが間違っていると「達成できない目標を達成できる」と表示してしまい、
実際の資金を失わせることになるので、検算まで含めて確認する。
"""

from __future__ import annotations

import math

import pytest

from llmfx.domain.risk import (
    RiskManager,
    expected_log_growth,
    kelly_fraction,
    monte_carlo,
    position_size,
    required_risk_fraction,
)


# ----------------------------------------------------------------------
def test_position_size_risks_exactly_the_requested_amount():
    units, risk = position_size(
        equity=10_000, risk_fraction=0.02, entry_price=150.0, stop_price=149.5
    )
    assert units == 400  # 200 / 0.5
    assert risk == pytest.approx(200.0)


def test_position_size_returns_zero_when_stop_is_at_entry():
    units, risk = position_size(10_000, 0.02, 150.0, 150.0)
    assert units == 0 and risk == 0


def test_position_size_scales_with_stop_distance():
    tight, _ = position_size(10_000, 0.02, 150.0, 149.9)
    wide, _ = position_size(10_000, 0.02, 150.0, 149.0)
    assert tight > wide, "損切りが浅いほど数量は大きくなる"


def test_position_size_respects_quote_conversion():
    base, _ = position_size(10_000, 0.02, 150.0, 149.5, quote_to_account_rate=1.0)
    converted, _ = position_size(10_000, 0.02, 150.0, 149.5, quote_to_account_rate=2.0)
    assert converted == pytest.approx(base / 2, rel=0.01)


# ----------------------------------------------------------------------
def test_kelly_is_negative_when_edge_is_negative():
    assert kelly_fraction(win_rate=0.3, win_r=1.0) < 0


def test_required_risk_fraction_actually_reaches_the_target():
    """逆算したリスク率で運用すると、本当に目標成長率になるかを検算する。"""
    for win_rate, win_r, trades in [(0.45, 2.5, 20), (0.55, 2.0, 30), (0.35, 3.0, 25)]:
        f = required_risk_fraction(1.4, trades, win_rate, win_r)
        assert f is not None
        growth = math.exp(trades * expected_log_growth(f, win_rate, win_r))
        assert growth == pytest.approx(1.4, rel=1e-6)


def test_unreachable_target_returns_none():
    """ケリー点でも届かない目標は None(=到達不能)を返すこと。"""
    # 期待値はわずかに正だが、月 5 トレードでは +40%/月 に届かない
    assert required_risk_fraction(1.4, trades_per_month=5, win_rate=0.35, win_r=2.0) is None


def test_negative_edge_is_unreachable_at_any_risk():
    assert required_risk_fraction(1.4, 30, win_rate=0.2, win_r=1.0) is None


def test_required_risk_never_exceeds_kelly():
    """必要リスク率がケリー点を超えることはない(超えると成長率が下がるため)。"""
    f = required_risk_fraction(1.4, 20, 0.45, 2.5)
    kelly = kelly_fraction(0.45, 2.5)
    assert f is not None and f <= kelly + 1e-9


def test_higher_target_requires_more_risk():
    low = required_risk_fraction(1.10, 20, 0.45, 2.5)
    high = required_risk_fraction(1.40, 20, 0.45, 2.5)
    assert low is not None and high is not None
    assert high > low


# ----------------------------------------------------------------------
def test_monte_carlo_ruin_probability_grows_with_risk():
    low = monte_carlo(0.02, 0.45, 2.5, trades_per_month=20, months=12, paths=4000)
    high = monte_carlo(0.25, 0.45, 2.5, trades_per_month=20, months=12, paths=4000)
    assert high.prob_ruin > low.prob_ruin
    assert high.median_max_drawdown > low.median_max_drawdown


def test_monte_carlo_is_deterministic_for_a_given_seed():
    a = monte_carlo(0.03, 0.45, 2.5, trades_per_month=20, paths=2000, seed=42)
    b = monte_carlo(0.03, 0.45, 2.5, trades_per_month=20, paths=2000, seed=42)
    assert a.prob_ruin == b.prob_ruin
    assert a.median_final_multiple == b.median_final_multiple


def test_expected_log_growth_is_negative_infinity_when_a_loss_wipes_the_account():
    assert expected_log_growth(1.0, 0.5, 2.0, loss_r=1.0) == float("-inf")


# ----------------------------------------------------------------------
def test_risk_manager_halts_on_max_drawdown():
    manager = RiskManager(initial_equity=10_000, max_drawdown_stop=0.20)
    manager.on_bar(10_000, day="d1")
    manager.on_bar(7_900, day="d1")
    assert manager.halted
    allowed, reason = manager.can_open(7_900)
    assert not allowed and reason


def test_risk_manager_blocks_new_entries_after_daily_loss_limit():
    manager = RiskManager(initial_equity=10_000, max_daily_loss=0.05, max_drawdown_stop=0.5)
    manager.on_bar(10_000, day="d1")
    manager.on_bar(9_400, day="d1")
    allowed, reason = manager.can_open(9_400)
    assert not allowed and "当日損失" in reason
    # 翌日になれば再開する
    manager.on_bar(9_400, day="d2")
    allowed, _ = manager.can_open(9_400)
    assert allowed


# --- 小数ロットとレバレッジ上限 ---------------------------------------------
# 暗号資産は 1 単位が数百万円になるため、整数へ丸めると建玉がすべて 0 になる。


def test_fractional_lots_are_not_rounded_away():
    units, risk = position_size(
        equity=1_000_000.0, risk_fraction=0.02,
        entry_price=7_000_000.0, stop_price=6_700_000.0,
        min_units=0.001, size_step=0.001,
    )
    # 20,000 / 300,000 = 0.0666... -> 0.001 刻みへ切り下げて 0.066
    assert units == pytest.approx(0.066)
    assert risk == pytest.approx(0.066 * 300_000)


def test_integer_sizing_still_applies_to_fx_defaults():
    units, _ = position_size(
        equity=10_000.0, risk_fraction=0.02, entry_price=150.0, stop_price=149.5
    )
    assert units == pytest.approx(400.0)


def test_size_below_the_minimum_order_is_rejected():
    units, risk = position_size(
        equity=10_000.0, risk_fraction=0.02,
        entry_price=7_000_000.0, stop_price=6_700_000.0,
        min_units=0.001, size_step=0.001,
    )
    assert units == 0.0 and risk == 0.0


def test_leverage_cap_limits_the_notional():
    """損切りが近いと建玉が資産の何倍にも膨らむ。国内暗号資産は 2 倍が上限。"""
    uncapped, _ = position_size(
        equity=1_000_000.0, risk_fraction=0.02,
        entry_price=7_000_000.0, stop_price=6_990_000.0,
        min_units=0.001, size_step=0.001,
    )
    capped, _ = position_size(
        equity=1_000_000.0, risk_fraction=0.02,
        entry_price=7_000_000.0, stop_price=6_990_000.0,
        min_units=0.001, size_step=0.001, max_leverage=2.0,
    )
    assert uncapped == pytest.approx(2.0)      # 1,400 万円 = 資産の 14 倍
    assert capped * 7_000_000.0 <= 1_000_000.0 * 2.0 + 1e-6
    assert capped == pytest.approx(0.285)


def test_leverage_cap_does_not_inflate_a_smaller_size():
    units, _ = position_size(
        equity=1_000_000.0, risk_fraction=0.02,
        entry_price=7_000_000.0, stop_price=6_700_000.0,
        min_units=0.001, size_step=0.001, max_leverage=2.0,
    )
    assert units == pytest.approx(0.066)
