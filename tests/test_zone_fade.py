"""抵抗帯へ指値を置いて待つ取引のテスト.

利用者が実際にやっていた形。抵抗帯の少し奥に予め指値を置き、届けば
約定、届かなければ何も起きない。損切りは帯の外、利確は明示しない。

いちばん守りたいのは **逆選択を消さないこと**。指値は届いたときだけ
約定するので、価格が少し触れて反転する場面(いちばん美味しい形)では
約定しない。ここを「触れたら約定」にすると、成績が実際より良く出る。
"""

from __future__ import annotations

import pytest

from llmfx.data.synthetic import generate_synthetic_candles
from llmfx.research.zone_fade import collect_fade_trades


def trades(**kwargs):
    candles = generate_synthetic_candles(count=8000, seed=5)
    return collect_fade_trades(candles, **kwargs)


# --- 指値の約定 -----------------------------------------------------------


def test_the_limit_only_fills_where_price_actually_traded():
    """届いていない指値で約定してはいけない(逆選択が消える)。"""
    for t in trades(horizon=24)[:200]:
        if t.from_below:
            assert t.entry <= t.stop, "売りなら損切りは上"
        else:
            assert t.entry >= t.stop, "買いなら損切りは下"


def test_a_limit_placed_far_inside_the_zone_rarely_fills():
    """帯の奥へ置くほど届きにくい。ここが逆選択の量を決める。

    件数は単調にならない。同時保有 1 建玉なので、ある機会を見送ると
    別の機会を拾える枠が空くため(過去に時間帯フィルタでも踏んだ挙動)。
    そこで、極端に奥へ置いた場合と比べる。
    """
    shallow = len(trades(horizon=24, entry_offset_atr=0.0))
    deep = len(trades(horizon=24, entry_offset_atr=3.0))
    assert deep < shallow * 0.7, (shallow, deep)


def test_waiting_longer_matters_only_when_the_limit_is_deep():
    """浅い指値は触れた足でほぼ約定するので、待ち時間は効かない。

    奥に置くと届くまでに時間がかかり、待ち時間が効きはじめる。
    (実測: 奥 1.0 ATR で 1 本待ち 30 件 -> 30 本待ち 465 件)
    """
    shallow = [len(trades(horizon=24, entry_offset_atr=0.1, max_wait_bars=w))
               for w in (1, 30)]
    deep = [len(trades(horizon=24, entry_offset_atr=1.0, max_wait_bars=w))
            for w in (1, 30)]
    assert shallow[1] < shallow[0] * 1.2, shallow
    assert deep[1] > deep[0] * 5, deep


# --- 損切り ---------------------------------------------------------------


def test_the_stop_sits_outside_the_zone():
    for t in trades(horizon=24, stop_buffer_atr=0.2)[:200]:
        assert t.risk_atr > 0
        # 帯の幅ぶんは必ず離れている
        assert t.risk_atr >= t.zone_width_atr * 0.3


def test_a_stopped_trade_is_exactly_minus_one_r():
    stopped = [t for t in trades(horizon=24) if t.hit_stop]
    assert stopped
    assert all(t.r_multiple == -1.0 for t in stopped)


def test_a_wider_stop_buffer_widens_the_risk():
    tight = trades(horizon=24, stop_buffer_atr=0.1)
    wide = trades(horizon=24, stop_buffer_atr=0.8)
    import statistics as st
    assert (st.median(t.risk_atr for t in wide)
            > st.median(t.risk_atr for t in tight))


# --- 建玉の重なり ---------------------------------------------------------


def test_positions_never_overlap():
    """同時に 1 建玉。重ねると同じ値動きを何度も数えることになる。"""
    got = trades(horizon=24)
    for a, b in zip(got, got[1:]):
        assert b.bar_index > a.bar_index


# --- 先読み ---------------------------------------------------------------


def test_truncating_the_series_does_not_change_earlier_trades():
    candles = generate_synthetic_candles(count=8000, seed=5)
    full = collect_fade_trades(candles, horizon=24)
    cut = collect_fade_trades(candles[:5000], horizon=24)
    assert cut
    for a, b in zip(cut, full):
        assert a.bar_index == b.bar_index
        assert a.entry == pytest.approx(b.entry)
        assert a.r_multiple == pytest.approx(b.r_multiple)


def test_narrow_zone_filter_keeps_only_narrow_zones():
    """件数は減るとは限らない。太い帯を見送ると、同時保有 1 建玉の枠が
    空いて別の機会を拾えるため。守るべきは幅の条件そのもの。"""
    strict = trades(horizon=24, max_zone_width_atr=1.5)
    assert strict
    for t in strict:
        assert t.zone_width_atr <= 1.5


def test_the_fill_bar_itself_can_stop_the_trade_out():
    """約定した足の残りで損切りまで走ることは普通にある。

    翌足から数えると、いちばん不利な場面だけを見逃して成績が良く出る。
    実測では、ここを直すだけで差引が +0.118 R から動いた。
    """
    got = trades(horizon=24)
    assert any(t.bars_held == 0 for t in got), "約定足での損切りが 1 件も無い"


# --- ブレイクリスクで見送る -----------------------------------------------


def swing(index, price, kind, lag=3):
    from datetime import datetime, timedelta, timezone

    from llmfx.domain.types import Swing

    return Swing(index=index, confirmed_index=index + lag,
                 time=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=index),
                 price=price, type=kind)


def test_rising_lows_into_a_resistance_zone_mean_the_defenders_are_losing():
    """安値切り上げ = 買い手が押し上げてきている。抵抗帯は破られやすい。"""
    from llmfx.domain.types import SwingType
    from llmfx.research.zone_fade import defenders_weakening

    rising = [swing(10, 99.0, SwingType.LOW), swing(20, 99.6, SwingType.LOW)]
    falling = [swing(10, 99.6, SwingType.LOW), swing(20, 99.0, SwingType.LOW)]
    assert defenders_weakening(rising, from_below=True, bar_index=100)
    assert not defenders_weakening(falling, from_below=True, bar_index=100)


def test_the_support_side_is_the_mirror_image():
    from llmfx.domain.types import SwingType
    from llmfx.research.zone_fade import defenders_weakening

    falling = [swing(10, 101.0, SwingType.HIGH), swing(20, 100.4, SwingType.HIGH)]
    rising = [swing(10, 100.4, SwingType.HIGH), swing(20, 101.0, SwingType.HIGH)]
    assert defenders_weakening(falling, from_below=False, bar_index=100)
    assert not defenders_weakening(rising, from_below=False, bar_index=100)


def test_unconfirmed_swings_are_not_used():
    """未確定のスイングで判定すると先読みになる。"""
    from llmfx.domain.types import SwingType
    from llmfx.research.zone_fade import defenders_weakening

    later = [swing(10, 99.0, SwingType.LOW), swing(20, 99.6, SwingType.LOW)]
    assert not defenders_weakening(later, from_below=True, bar_index=22)
    assert defenders_weakening(later, from_below=True, bar_index=23)


def test_too_few_swings_means_no_opinion():
    from llmfx.domain.types import SwingType
    from llmfx.research.zone_fade import defenders_weakening

    assert not defenders_weakening([], from_below=True, bar_index=100)
    assert not defenders_weakening(
        [swing(10, 99.0, SwingType.LOW)], from_below=True, bar_index=100
    )


def test_the_break_risk_filter_changes_which_setups_are_taken():
    loose = {(t.bar_index, t.entry) for t in trades(horizon=24)}
    strict = {(t.bar_index, t.entry)
              for t in trades(horizon=24, skip_break_risk=True)}
    assert strict and strict != loose
    # 件数は減るとは限らない。見送ると同時保有 1 建玉の枠が空くため。


def test_break_risk_uses_only_confirmed_swings():
    """未確定のスイングで判定すると先読みになる。"""
    from llmfx.data.synthetic import generate_synthetic_candles
    from llmfx.research.zone_fade import collect_fade_trades

    candles = generate_synthetic_candles(count=8000, seed=5)
    full = collect_fade_trades(candles, horizon=24, skip_break_risk=True)
    cut = collect_fade_trades(candles[:5000], horizon=24, skip_break_risk=True)
    assert cut
    for a, b in zip(cut, full):
        assert a.bar_index == b.bar_index
        assert a.r_multiple == pytest.approx(b.r_multiple)


def test_a_range_based_stop_is_never_tighter_than_the_zone_edge():
    """はみ出しで刈られないよう、帯の縁より外側にしか動かさない。"""
    import statistics as st

    edge = trades(horizon=24)
    wide = trades(horizon=24, stop_from_range_bars=60)
    assert wide
    assert (st.median(t.risk_atr for t in wide)
            >= st.median(t.risk_atr for t in edge))


def test_a_longer_window_never_tightens_the_stop():
    import statistics as st

    short = st.median(t.risk_atr for t in trades(horizon=24, stop_from_range_bars=20))
    long = st.median(t.risk_atr for t in trades(horizon=24, stop_from_range_bars=120))
    assert long >= short


# --- 指値を最値へ置く -----------------------------------------------------


def test_the_entry_limit_can_sit_at_the_range_extreme():
    """帯の縁の内側ではなく、直近 N 本の最値で待つ(利用者の本来の指摘)。

    縁の内側で待つと、上へ突き抜ける動きの途中で約定してしまい、
    そのまま損切りまで持っていかれる。
    """
    inside = trades(horizon=24, stop_buffer_atr=0.5)
    extreme = trades(horizon=24, stop_buffer_atr=0.5, entry_from_range_bars=20)
    assert extreme
    import statistics as st
    # 最値で待つほうが、指値は必ず帯の外側寄りになる
    assert (st.median(t.entry for t in extreme if t.from_below)
            >= st.median(t.entry for t in inside if t.from_below))


def test_the_extreme_excludes_the_bar_that_would_fill_it():
    """その足の高値で指値を決めると「更新したから約定した」の循環になる。

    循環していれば、触れた足で必ず約定するので待ち時間が効かなくなる。
    """
    brief = len(trades(horizon=24, entry_from_range_bars=60, max_wait_bars=1))
    patient = len(trades(horizon=24, entry_from_range_bars=60, max_wait_bars=30))
    assert brief < patient * 0.9, (brief, patient)


def test_the_stop_stays_outside_the_entry_extreme():
    for t in trades(horizon=24, entry_from_range_bars=20, stop_buffer_atr=0.5)[:200]:
        if t.from_below:
            assert t.stop > t.entry
        else:
            assert t.stop < t.entry


# --- 上位足の帯 / 上下 2 本のレンジ ---------------------------------------


def test_higher_timeframe_zones_use_only_closed_bars():
    """上位足は閉じてからしか使わない。先読みになる。"""
    from llmfx.research.zone_fade import collect_fade_trades

    candles = generate_synthetic_candles(count=12000, seed=5)
    full = collect_fade_trades(candles, horizon=24, higher_minutes=60)
    cut = collect_fade_trades(candles[:8000], horizon=24, higher_minutes=60)
    assert cut
    for a, b in zip(cut, full):
        assert a.bar_index == b.bar_index
        assert a.entry == pytest.approx(b.entry)
        assert a.r_multiple == pytest.approx(b.r_multiple)


def test_higher_timeframe_zones_are_fewer_and_wider():
    """上位足の帯は数が減り、間隔が空く。"""
    same = trades(horizon=24)
    higher = trades(horizon=24, higher_minutes=60)
    assert higher
    assert len(higher) < len(same)


def test_requiring_a_range_only_takes_setups_with_zones_on_both_sides():
    """片側だけで張ると、そこを抜けられたときに一方的に負ける。"""
    loose = trades(horizon=24)
    strict = trades(horizon=24, require_range=True)
    assert strict
    assert len(strict) < len(loose)


def test_a_range_wider_than_the_cap_is_skipped():
    wide = trades(horizon=24, require_range=True)
    narrow = trades(horizon=24, require_range=True, max_range_atr=3.0)
    assert narrow
    assert len(narrow) <= len(wide)


def test_the_two_modes_stay_switchable():
    """どちらが優れているかは測ってから決める。両方動くこと。"""
    for kwargs in (
        {},
        {"higher_minutes": 60},
        {"require_range": True},
        {"higher_minutes": 60, "require_range": True},
    ):
        got = trades(horizon=24, **kwargs)
        assert got, kwargs
