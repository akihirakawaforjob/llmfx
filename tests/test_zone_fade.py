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


def test_narrow_zone_filter_only_removes_trades():
    loose = trades(horizon=24)
    strict = trades(horizon=24, max_zone_width_atr=1.5)
    assert len(strict) <= len(loose)
    for t in strict:
        assert t.zone_width_atr <= 1.5
