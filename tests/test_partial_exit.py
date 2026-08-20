"""分割決済(資金管理の一階)のテスト.

実測では、負けの 34.6% が一度 +1.0R まで順行してから -1.0R で切られている。
入る場所ではなく持ち方で失っているので、決済の型で拾える余地がある。

いちばん守りたいのは **同じ足で損切りにも触れていたら成立させない** こと。
足の中の順序は分からないので、損切りが先という前提を崩してはいけない。
ここを緩めると、都合の良い順序を仮定して数字を良く見せることになる。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llmfx.config import AppConfig
from llmfx.domain.types import Candle, Side, Signal, Position
from llmfx.execution.fills import apply_partial_exit, evaluate_partial_exit

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def config(**execution) -> AppConfig:
    base = {"partial_exit_at_r": 1.0, "partial_exit_fraction": 0.5}
    base.update(execution)
    cfg = AppConfig.from_dict({"execution": base})
    cfg.validate()
    return cfg


def position(side: Side = Side.LONG, entry: float = 100.0, stop: float = 99.0) -> Position:
    signal = Signal(
        time=T0, bar_index=0, side=side, reference_price=entry,
        stop_loss=stop, take_profit=entry + 10, risk_per_unit=abs(entry - stop),
        reward_per_unit=10.0, rr=10.0, broken_level=entry, stop_basis=stop,
        target_source="test", structure=None, reason="test",
    )
    return Position(
        signal=signal, side=side, units=1000.0, entry_price=entry,
        entry_time=T0, entry_index=0, stop_loss=stop, take_profit=entry + 10,
        initial_risk_per_unit=abs(entry - stop), risk_amount=1000.0,
    )


def bar(o, h, l, c, i=1) -> Candle:
    return Candle(time=T0 + timedelta(hours=i), open=o, high=h, low=l, close=c)


# --- 成立する場合 ---------------------------------------------------------


def test_reaching_the_level_fills_at_the_level_not_worse():
    """指値なのでスリッページを乗せない。"""
    assert evaluate_partial_exit(position(), bar(100, 101.5, 99.8, 101), config()) == 101.0


def test_not_reaching_the_level_does_nothing():
    assert evaluate_partial_exit(position(), bar(100, 100.5, 99.8, 100.2), config()) is None


def test_half_the_position_is_closed_and_the_rest_is_kept():
    pos = position()
    cfg = config(partial_exit_fraction=0.5)
    pnl = apply_partial_exit(pos, 101.0, cfg)
    assert pos.units == pytest.approx(500.0)
    assert pos.scaled_units == pytest.approx(500.0)
    assert pnl == pytest.approx(1.0 * 500.0), "1.0 の値幅 x 500 通貨"
    assert pos.scaled_out


def test_it_only_happens_once_per_position():
    pos = position()
    cfg = config()
    apply_partial_exit(pos, 101.0, cfg)
    assert evaluate_partial_exit(pos, bar(101, 103, 100.5, 102), cfg) is None


# --- 同じ足で損切りにも触れた場合 -----------------------------------------


def test_a_bar_that_also_touches_the_stop_does_not_scale_out():
    """順序が分からない以上、損切りが先に約定した扱いにする。"""
    touched = bar(100, 101.5, 98.9, 99.5)     # 高値は +1R、安値は損切り
    assert evaluate_partial_exit(position(), touched, config()) is None


def test_a_gap_through_the_stop_does_not_scale_out():
    gapped = bar(98.5, 101.5, 98.0, 101.0)    # 始値で既に損切りの下
    assert evaluate_partial_exit(position(), gapped, config()) is None


# --- ショート -------------------------------------------------------------


def test_the_short_side_is_the_mirror_image():
    pos = position(side=Side.SHORT, entry=100.0, stop=101.0)
    cfg = config()
    assert evaluate_partial_exit(pos, bar(100, 100.2, 98.5, 99), cfg) == 99.0
    pnl = apply_partial_exit(pos, 99.0, cfg)
    assert pnl == pytest.approx(1.0 * 500.0), "売りでも下がれば利益"


def test_a_short_bar_touching_its_stop_does_not_scale_out():
    pos = position(side=Side.SHORT, entry=100.0, stop=101.0)
    assert evaluate_partial_exit(pos, bar(100, 101.2, 98.5, 99), config()) is None


# --- 残玉を建値へ移す -----------------------------------------------------


def test_the_remainder_can_be_moved_to_break_even():
    pos = position()
    apply_partial_exit(pos, 101.0, config(break_even_after_partial=True))
    assert pos.stop_loss == pytest.approx(100.0)
    assert pos.moved_to_break_even


def test_the_remainder_is_left_alone_by_default():
    pos = position()
    apply_partial_exit(pos, 101.0, config())
    assert pos.stop_loss == pytest.approx(99.0)


def test_break_even_never_moves_the_stop_backwards():
    """既に建値より上へ動かしてあるなら、下げてはいけない。"""
    pos = position()
    pos.stop_loss = 100.5
    apply_partial_exit(pos, 101.0, config(break_even_after_partial=True))
    assert pos.stop_loss == pytest.approx(100.5)


# --- 設定 -----------------------------------------------------------------


def test_it_is_off_unless_asked_for():
    cfg = AppConfig.from_dict({})
    assert cfg.execution.partial_exit_at_r is None
    assert evaluate_partial_exit(position(), bar(100, 105, 99.8, 104), cfg) is None


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 1.5])
def test_a_fraction_outside_zero_and_one_is_rejected(fraction):
    from llmfx.config import ConfigError

    with pytest.raises(ConfigError):
        AppConfig.from_dict(
            {"execution": {"partial_exit_at_r": 1.0, "partial_exit_fraction": fraction}}
        ).validate()
