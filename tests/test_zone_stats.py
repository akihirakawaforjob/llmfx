"""帯への接触を数える観測のテスト.

戦略ではなく観測だが、先読みが混ざれば観測そのものが無意味になる。
バックテストと同じ性質を要求する:

  データを途中で打ち切っても、それ以前の事象が 1 件も変わらないこと
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llmfx.data.synthetic import generate_synthetic_candles
from llmfx.domain.types import Candle
from llmfx.research.zone_stats import (
    bucket_by_defence,
    bucket_by_touches,
    collect_touches,
)

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def wave(levels: list[float], per_leg: int) -> list[Candle]:
    """折れ点をつないだ、きれいな三角波を作る。"""
    out: list[Candle] = []
    price = levels[0]
    for a, b in zip(levels, levels[1:]):
        step = (b - a) / per_leg
        for _ in range(per_leg):
            nxt = price + step
            out.append(
                Candle(
                    time=T0 + timedelta(hours=len(out)),
                    open=price,
                    high=max(price, nxt) + 0.02,
                    low=min(price, nxt) - 0.02,
                    close=nxt,
                    volume=1.0,
                )
            )
            price = nxt
    return out


# --- 先読み ---------------------------------------------------------------


def test_truncating_the_data_does_not_change_earlier_events():
    """これが崩れたら、観測している数字に意味が無い。"""
    candles = generate_synthetic_candles(count=3000, seed=11)
    full = collect_touches(candles, horizon=12)
    cut = collect_touches(candles[:2000], horizon=12)
    assert cut, "検証できるだけの事象が出ていること"

    limit = 2000 - 12 - 1
    a = [(e.bar_index, e.touches, round(e.fade_move, 9))
         for e in full if e.bar_index < limit]
    b = [(e.bar_index, e.touches, round(e.fade_move, 9))
         for e in cut if e.bar_index < limit]
    assert a == b, "打ち切ると過去の事象が変わっている = 先読み"


def test_the_forward_window_starts_after_the_decision_bar():
    """判定に使った足を成果に含めると、その足の値動きを二度使うことになる。"""
    candles = generate_synthetic_candles(count=1200, seed=3)
    events = collect_touches(candles, horizon=1, confirm=3)
    assert events
    for e in events[:50]:
        assert e.bar_index <= e.decision_index < e.bar_index + 3
        start = candles[e.decision_index].close
        nxt = candles[e.decision_index + 1]
        sign = -1.0 if e.from_below else 1.0
        expected = (nxt.close - start) * sign / e.atr
        assert e.fade_move == pytest.approx(expected)


def test_the_direction_of_approach_uses_the_bar_before_the_touch():
    """触れた足の終値で決めると、突き抜けた足が「上から来た」判定に化ける。"""
    candles = generate_synthetic_candles(count=3000, seed=8)
    events = collect_touches(candles, horizon=6, confirm=3)
    assert events
    for e in events[:100]:
        prior = candles[e.bar_index - 1].close
        assert e.from_below == (prior < e.zone_price)


def test_a_break_and_a_rejection_are_not_pooled_together():
    """混ぜて平均すると打ち消し合ってきっかり 0 になる(実データでそうなった)。"""
    from llmfx.research.zone_stats import bucket_by_reaction

    candles = generate_synthetic_candles(count=6000, seed=5)
    events = collect_touches(candles, horizon=12, confirm=3)
    labels = {b.label for b in bucket_by_reaction(events)}
    assert {"弾かれた", "抜けた"} <= labels, labels
    for b in bucket_by_reaction(events):
        assert b.count > 0


def test_events_are_never_emitted_without_a_full_forward_window():
    candles = generate_synthetic_candles(count=1500, seed=9)
    events = collect_touches(candles, horizon=30)
    assert events
    assert max(e.bar_index for e in events) + 30 < len(candles)


# --- 数え方 ---------------------------------------------------------------


def test_a_zone_is_counted_once_per_visit_not_once_per_bar():
    """帯に張り付いている間ずっと数えると、回数が水増しされる。"""
    candles = generate_synthetic_candles(count=3000, seed=7)
    loose = collect_touches(candles, horizon=12, rearm_atr=5.0)
    tight = collect_touches(candles, horizon=12, rearm_atr=0.05)
    assert len(loose) < len(tight), "離れる条件を厳しくしても事象が減っていない"


def test_zones_below_the_touch_threshold_produce_no_events():
    candles = generate_synthetic_candles(count=2000, seed=4)
    many = collect_touches(candles, horizon=12, min_touches=2)
    few = collect_touches(candles, horizon=12, min_touches=8)
    assert len(few) < len(many)


def test_a_clean_bounce_is_recorded_as_a_positive_fade():
    """同じ天井を 3 回叩いて落ちる形。跳ね返り方向が正になること。"""
    candles = wave([100, 110, 101, 110, 101, 110, 96], per_leg=14)
    events = collect_touches(
        candles, horizon=10, warmup=20, min_touches=2, tolerance_atr=1.5
    )
    assert events, "天井を叩いた事象が拾えていない"
    top = [e for e in events if e.from_below]
    assert top, "抵抗として試した事象が無い"
    assert sum(e.fade_move for e in top) / len(top) > 0, "落ちたのに負で記録されている"


# --- まとめ方 -------------------------------------------------------------


def test_buckets_cover_every_event_exactly_once():
    candles = generate_synthetic_candles(count=3000, seed=5)
    events = collect_touches(candles, horizon=12)
    assert sum(b.count for b in bucket_by_touches(events)) == len(events)
    assert sum(b.count for b in bucket_by_defence(events)) == len(events)


def test_no_events_gives_no_buckets():
    assert bucket_by_touches([]) == []
    assert bucket_by_defence([]) == []


# --- 標本の重なり ---------------------------------------------------------


def test_one_per_bar_keeps_only_the_nearest_zone():
    """1 本の足が複数の帯に触れると、同じ値動きを何度も数えることになる。"""
    candles = generate_synthetic_candles(count=3000, seed=5)
    everything = collect_touches(candles, horizon=12)
    single = collect_touches(candles, horizon=12, one_per_bar=True)
    assert len(single) < len(everything)
    bars = [e.bar_index for e in single]
    assert len(bars) == len(set(bars)), "同じ足から 2 件以上採っている"


def test_cooldown_stops_the_forward_windows_from_overlapping():
    """窓が重なった標本を独立として数えると、t 値が大きく出すぎる。"""
    candles = generate_synthetic_candles(count=3000, seed=5)
    events = collect_touches(candles, horizon=20, one_per_bar=True, cooldown=20)
    assert events
    gaps = [b - a for a, b in zip(
        [e.bar_index for e in events], [e.bar_index for e in events][1:]
    )]
    assert all(g >= 20 for g in gaps), f"窓が重なっている: 最小 {min(gaps)}"


def test_cooldown_does_not_invent_events():
    candles = generate_synthetic_candles(count=3000, seed=5)
    loose = {(e.bar_index, round(e.zone_price, 6))
             for e in collect_touches(candles, horizon=20, one_per_bar=True)}
    strict = {(e.bar_index, round(e.zone_price, 6))
              for e in collect_touches(candles, horizon=20, one_per_bar=True, cooldown=20)}
    assert strict <= loose, "元に無い事象が生えている"


# --- 上位足で帯を見つけ、下位足で測る -------------------------------------


def two_timeframes(count: int = 12000, seed: int = 5):
    from llmfx.data.resample import resample_candles

    lower = generate_synthetic_candles(count=count, seed=seed)
    return resample_candles(lower, 60), lower


def test_a_higher_timeframe_bar_is_not_used_before_it_closes():
    """時刻 T の上位足は T+60 分になって初めて参照してよい。

    ここを緩めると、下位足で見ているのに上位足の未確定バーを覗くことになり、
    測っている数字が丸ごと嘘になる。
    """
    from llmfx.research.zone_stats import collect_touches_mtf

    higher, lower = two_timeframes()
    full = collect_touches_mtf(higher, lower, higher_minutes=60, horizon=24)
    assert full, "検証できるだけの事象が出ていること"

    # 上位足を後ろから 20 本削る。削った範囲が閉じる前の事象は変わらないはず。
    cut_at = higher[-20].time
    trimmed = collect_touches_mtf(
        higher[:-20], lower, higher_minutes=60, horizon=24
    )
    before = [
        (e.bar_index, e.decision_index, round(e.follow_move, 9))
        for e in full
        if lower[e.bar_index].time < cut_at
    ]
    after = [
        (e.bar_index, e.decision_index, round(e.follow_move, 9))
        for e in trimmed
        if lower[e.bar_index].time < cut_at
    ]
    assert before == after, "上位足の未来を覗いている"


def test_truncating_the_lower_series_does_not_change_earlier_events():
    from llmfx.research.zone_stats import collect_touches_mtf

    higher, lower = two_timeframes()
    full = collect_touches_mtf(higher, lower, higher_minutes=60, horizon=24)
    cut = collect_touches_mtf(higher, lower[:8000], higher_minutes=60, horizon=24)
    assert cut

    limit = 8000 - 24 - 3 - 1
    a = [(e.bar_index, round(e.follow_move, 9)) for e in full if e.bar_index < limit]
    b = [(e.bar_index, round(e.follow_move, 9)) for e in cut if e.bar_index < limit]
    assert a == b, "打ち切ると過去の事象が変わっている = 先読み"


def test_the_forward_window_never_runs_past_the_end():
    from llmfx.research.zone_stats import collect_touches_mtf

    higher, lower = two_timeframes()
    events = collect_touches_mtf(higher, lower, higher_minutes=60, horizon=48)
    assert events
    assert max(e.decision_index for e in events) + 48 < len(lower)


def test_zone_width_is_recorded_so_the_stop_distance_is_known():
    """損切りは帯の外に置くので、帯の幅がそのままリスク幅の下限になる。"""
    from llmfx.research.zone_stats import collect_touches_mtf

    higher, lower = two_timeframes()
    events = collect_touches_mtf(higher, lower, higher_minutes=60, horizon=24)
    assert events
    assert all(e.zone_width_atr > 0 for e in events)


def test_one_per_bar_is_on_by_default_for_the_two_timeframe_version():
    """下位足では同じ帯に何度も触れるので、既定で絞っておく。"""
    from llmfx.research.zone_stats import collect_touches_mtf

    higher, lower = two_timeframes()
    events = collect_touches_mtf(higher, lower, higher_minutes=60, horizon=24)
    bars = [e.bar_index for e in events]
    assert len(bars) == len(set(bars))
