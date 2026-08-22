"""建玉の解剖のテスト.

利用者の切り分け:

    買いで入り最高値ではプラスだったのに最終的にマイナスなら、決済の位置に
    誤りがある。逆に直ぐに損切られているなら、損切り位置かエントリー位置が
    おかしい。

**群ごとに打つ手が真逆になる**ので、ここを取り違えると直す場所を間違える。
"""

from __future__ import annotations

from llmfx.data.synthetic import generate_synthetic_candles
from llmfx.research.postmortem import (
    diagnose, features, group_table, monotonic, quantiles,
)
from llmfx.research.zone_fade import collect_fade_trades


def trades_and_rows():
    cs = generate_synthetic_candles(count=20_000, seed=5)
    ts = collect_fade_trades(
        cs, stop_buffer_atr=1.5, max_wait_bars=12, horizon=240, higher_minutes=60,
        zone_source="range", range_bars=120, entry_at_zone_extreme=True,
        edge_mode="fade", exit_at_opposite_zone=True, max_open=4)
    return cs, ts, features(cs, ts)


# --- 切り分け -------------------------------------------------------------


def test_a_loser_that_ran_our_way_is_an_exit_problem():
    assert diagnose(-1.0, 2.4, 1.0) == "伸びたのに取れなかった"


def test_a_loser_that_never_moved_is_an_entry_problem():
    assert diagnose(-1.0, 0.1, 1.0) == "一度も伸びなかった"


def test_a_winner_is_a_winner_however_it_got_there():
    assert diagnose(1.2, 0.2, 0.9) == "取れた"


def test_the_middle_is_named_and_not_forced_into_a_group():
    """0.5〜1.0R しか伸びなかった負けは、どちらとも言えない。

    無理にどちらかへ入れると、打つ手を取り違える。
    """
    assert diagnose(-1.0, 0.7, 1.0) == "どちらとも言えない"


# --- 特徴 -----------------------------------------------------------------


def test_features_only_use_the_past():
    """**先の値動きを混ぜない。**打ち切っても特徴が変わらないこと。"""
    cs, ts, rows = trades_and_rows()
    cut = max(t.fill_index for t in ts[:40]) + 5
    part = collect_fade_trades(
        cs[:cut], stop_buffer_atr=1.5, max_wait_bars=12, horizon=240,
        higher_minutes=60, zone_source="range", range_bars=120,
        entry_at_zone_extreme=True, edge_mode="fade", exit_at_opposite_zone=True,
        max_open=4)
    prows = features(cs[:cut], part)
    assert prows
    by_full = {r.at: r for r in rows}
    checked = 0
    for r in prows[:40]:
        if r.at not in by_full:
            continue
        for name in ("直線性(96本)", "直近の動き(24本 ATR・順方向+)", "UTC 時"):
            assert abs(r.f[name] - by_full[r.at].f[name]) < 1e-6, name
        checked += 1
    assert checked > 3, checked


def test_every_row_carries_the_excursions():
    """最高値・最安値を残す。ここが無いと切り分けができない。"""
    _, ts, rows = trades_and_rows()
    assert rows
    for r in rows:
        assert r.mfe >= 0 and r.mae >= 0


def test_groups_cover_every_trade():
    _, _, rows = trades_and_rows()
    assert sum(g[1] for g in group_table(rows)) == len(rows)


# --- 分位 -----------------------------------------------------------------


def test_quantiles_split_evenly_and_report_expectancy():
    _, _, rows = trades_and_rows()
    t = quantiles(rows, "直線性(96本)", n=5)
    assert len(t) == 5
    sizes = [n for _, _, n, _ in t]
    assert max(sizes) - min(sizes) <= 1
    # 区分は昇順に並ぶ
    assert all(a[0] <= b[0] for a, b in zip(t, t[1:]))


def test_quantiles_refuse_a_sample_too_small_to_split():
    _, _, rows = trades_and_rows()
    assert quantiles(rows[:30], "直線性(96本)", n=5) == []


def test_monotonic_scores_a_straight_line_at_one():
    fake = [(0, 1, 10, -0.2), (1, 2, 10, 0.0), (2, 3, 10, 0.3), (3, 4, 10, 0.6)]
    assert monotonic(fake) == 1.0


def test_monotonic_scores_a_zigzag_low():
    fake = [(0, 1, 10, 0.5), (1, 2, 10, -0.5), (2, 3, 10, 0.5), (3, 4, 10, -0.5)]
    assert monotonic(fake) < 0.7
