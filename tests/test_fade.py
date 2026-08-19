"""ダマシを取る逆張り(fade)のテスト.

FX 8 銘柄 x 20 年 22,264 件で、順張りは -0.114 R(t=-7.68)、コストを外しても
-0.071 R(t=-5.33)だった。値動きそのものがブレイク方向と逆に偏っている。
その偏りを取りにいくのが fade。

順張りとは損切り・利確の置き方が根本的に違うので、単純な符号反転で
ないことを固定する。
"""

from __future__ import annotations

import pytest

from llmfx.config import AppConfig, ConfigError
from llmfx.data.synthetic import generate_synthetic_candles
from llmfx.domain.strategy import DowReversalStrategy
from llmfx.domain.types import Side


def run(mode: str, **entry_over):
    candles = generate_synthetic_candles(count=8000, seed=20260810)
    config = AppConfig()
    config.entry.mode = mode
    config.entry.min_rr = 1e-6
    for key, value in entry_over.items():
        setattr(config.entry, key, value)
    config.validate()
    strategy = DowReversalStrategy(config)
    return [s for c in candles if (s := strategy.update(c))], strategy


# --- 設定 -----------------------------------------------------------------


def test_default_mode_is_breakout():
    """既存の挙動を変えない。fade は明示的に選ぶ。"""
    assert AppConfig().entry.mode == "breakout"


def test_unknown_mode_is_rejected():
    config = AppConfig()
    config.entry.mode = "reverse"
    with pytest.raises(ConfigError, match="entry.mode"):
        config.validate()


def test_non_positive_fade_parameters_are_rejected():
    for field in ("fade_target_r", "fade_stop_buffer_atr"):
        config = AppConfig()
        setattr(config.entry, field, 0.0)
        with pytest.raises(ConfigError, match=field):
            config.validate()


# --- 向き -----------------------------------------------------------------


def test_fade_takes_the_opposite_side_of_the_break():
    """上抜けなら売り、下抜けなら買い。"""
    breakout, _ = run("breakout")
    fade, _ = run("fade")
    assert breakout and fade
    by_time = {s.time: s for s in breakout}
    compared = 0
    for signal in fade:
        other = by_time.get(signal.time)
        if other is None:
            continue
        assert signal.side is not other.side, "同じ向きになっている"
        compared += 1
    assert compared > 0, "比較できるシグナルが無い"


# --- 損切りと利確の置き方 --------------------------------------------------


def test_stop_sits_beyond_the_break_extreme():
    """本物のブレイクならすぐ切れる位置。ここが遠いと逆張りの利点が消える。"""
    fade, _ = run("fade")
    for s in fade:
        if s.side is Side.SHORT:
            assert s.stop_loss > s.reference_price
            assert s.take_profit < s.reference_price
        else:
            assert s.stop_loss < s.reference_price
            assert s.take_profit > s.reference_price


def test_target_is_a_fixed_multiple_of_risk():
    """遠い目標を狙うと勝率が落ちて元の木阿弥。R 倍数で固定する。"""
    for r in (0.5, 1.0, 2.0):
        fade, _ = run("fade", fade_target_r=r)
        assert fade
        for s in fade:
            assert s.rr == pytest.approx(r, abs=1e-9)
            assert abs(s.take_profit - s.reference_price) == pytest.approx(
                s.risk_per_unit * r, rel=1e-9
            )


def test_wider_stop_buffer_makes_the_risk_larger():
    tight, _ = run("fade", fade_stop_buffer_atr=0.1)
    wide, _ = run("fade", fade_stop_buffer_atr=1.0)
    assert tight and wide
    tight_by_time = {s.time: s for s in tight}
    pairs = [(tight_by_time[s.time], s) for s in wide if s.time in tight_by_time]
    assert pairs, "比較できるシグナルが無い"
    assert sum(1 for a, b in pairs if b.risk_per_unit > a.risk_per_unit) > len(pairs) * 0.9


def test_fade_risk_is_much_smaller_than_breakout_risk():
    """順張りは転換前の極値まで、逆張りはブレイク極値の少し外まで。"""
    breakout, _ = run("breakout")
    fade, _ = run("fade")
    b_by_time = {s.time: s for s in breakout}
    pairs = [(b_by_time[s.time], s) for s in fade if s.time in b_by_time]
    assert pairs
    smaller = sum(1 for b, f in pairs if f.risk_per_unit < b.risk_per_unit)
    assert smaller > len(pairs) * 0.8, "逆張りの損切りが小さくなっていない"


def test_min_rr_does_not_filter_fade_signals():
    """fade の RR は fade_target_r で固定されるので、min_rr で選別しない。"""
    candles = generate_synthetic_candles(count=8000, seed=20260810)
    config = AppConfig()
    config.entry.mode = "fade"
    config.entry.fade_target_r = 1.0
    config.entry.min_rr = 3.0  # 順張りならほぼ全部落ちる水準
    config.validate()
    strategy = DowReversalStrategy(config)
    kept = [s for c in candles if (s := strategy.update(c))]
    assert kept, "min_rr が fade にも効いてしまっている"
    assert not any(r.reason == "rr_below_minimum" for r in strategy.rejections)


def test_shared_filters_still_apply_in_fade_mode():
    """時間帯フィルタなど、共通のフィルタは fade でも効く。"""
    wide, _ = run("fade")
    narrow, strategy = run("fade", allowed_hours_utc=[[7, 16]])
    assert len(narrow) < len(wide)
    assert all(7 <= s.time.hour < 16 for s in narrow)
    assert any(r.reason == "outside_session" for r in strategy.rejections)
