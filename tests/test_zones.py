"""抵抗帯(何度も試された水準)を数える仕組みのテスト.

これまでの実装が抜けていたのは 1 回しか触られていないスイング高値で、
そこを抜けた瞬間に順張りするとコストゼロでも負けた。利用者の見立ては
「そこを守っている側がいる」というもの。

守りたいのは 3 つ:
  先読みをしないこと    未確定のスイングを帯に入れない
  物差しが銘柄で狂わないこと  同じ水準の判定は ATR 比で持つ
  回数を水増ししないこと  同じスイングを二度数えない
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llmfx.domain.types import Swing, SwingType
from llmfx.domain.zones import ZoneTracker

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def swing(index: int, price: float, kind: SwingType = SwingType.HIGH,
          lag: int = 3) -> Swing:
    return Swing(index=index, confirmed_index=index + lag,
                 time=T0 + timedelta(hours=index), price=price, type=kind)


def feed(tracker: ZoneTracker, swings: list[Swing], atr: float = 1.0) -> None:
    """それぞれが確定したバーで取り込む。"""
    for s in swings:
        tracker.update(s, atr=atr, bar_index=s.confirmed_index)


# --- 先読み ---------------------------------------------------------------


def test_a_swing_is_ignored_until_the_bar_it_confirms_on():
    tracker = ZoneTracker()
    s = swing(10, 100.0, lag=3)
    assert tracker.update(s, atr=1.0, bar_index=12) is None, "確定前に取り込んでいる"
    assert tracker.update(s, atr=1.0, bar_index=13) is not None


def test_zones_are_hidden_until_their_last_touch_confirms():
    tracker = ZoneTracker()
    feed(tracker, [swing(10, 100.0), swing(20, 100.2)])
    assert tracker.zones(bar_index=22) == [], "まだ確定していない帯が見えている"
    assert len(tracker.zones(bar_index=23)) == 1


# --- まとめ方 -------------------------------------------------------------


def test_nearby_swings_land_in_the_same_zone():
    tracker = ZoneTracker(tolerance_atr=0.5)
    feed(tracker, [swing(10, 100.0), swing(20, 100.3), swing(30, 99.8)])
    zones = tracker.zones(bar_index=100)
    assert len(zones) == 1
    assert zones[0].count == 3


def test_far_apart_swings_make_separate_zones():
    tracker = ZoneTracker(tolerance_atr=0.5)
    feed(tracker, [swing(10, 100.0), swing(20, 108.0)])
    assert len(tracker.zones(bar_index=100)) == 2


def test_the_tolerance_scales_with_atr_not_with_price():
    """値幅の大きい銘柄では、同じ pips でも「同じ水準」ではない。"""
    tight = ZoneTracker(tolerance_atr=0.5)
    feed(tight, [swing(10, 100.0), swing(20, 100.4)], atr=1.0)
    assert len(tight.zones(bar_index=100)) == 1

    loose = ZoneTracker(tolerance_atr=0.5)
    feed(loose, [swing(10, 100.0), swing(20, 100.4)], atr=0.1)
    assert len(loose.zones(bar_index=100)) == 2, "ATR が小さければ別の水準"


def test_the_same_swing_is_never_counted_twice():
    tracker = ZoneTracker()
    s = swing(10, 100.0)
    tracker.update(s, atr=1.0, bar_index=13)
    tracker.update(s, atr=1.0, bar_index=14)
    tracker.update(s, atr=1.0, bar_index=15)
    assert tracker.zones(bar_index=100)[0].count == 1


def test_a_zone_remembers_which_side_tested_it():
    """抜けた抵抗は支持に変わる、と言われる。両側から試された帯を見分ける。"""
    tracker = ZoneTracker()
    feed(tracker, [swing(10, 100.0, SwingType.HIGH),
                   swing(20, 100.2, SwingType.LOW)])
    zone = tracker.zones(bar_index=100)[0]
    assert zone.sides == {SwingType.HIGH, SwingType.LOW}


# --- 絞り込み -------------------------------------------------------------


def test_zones_below_the_touch_threshold_are_not_returned():
    tracker = ZoneTracker()
    feed(tracker, [swing(10, 100.0), swing(20, 100.2), swing(30, 120.0)])
    twice = tracker.zones(bar_index=100, min_touches=2)
    assert len(twice) == 1 and twice[0].count == 2


def test_stale_zones_drop_out():
    tracker = ZoneTracker(max_age_bars=50)
    feed(tracker, [swing(10, 100.0), swing(20, 100.2)])
    assert tracker.zones(bar_index=60), "まだ生きているはず"
    assert not tracker.zones(bar_index=200), "古い帯が残っている"


def test_nearest_picks_the_closest_qualifying_zone():
    tracker = ZoneTracker()
    feed(tracker, [swing(10, 100.0), swing(20, 100.2),
                   swing(30, 120.0), swing(40, 120.1)])
    near = tracker.nearest(price=101.0, bar_index=100, min_touches=2)
    assert near is not None and near.price == pytest.approx(100.1)


def test_nearest_ignores_zones_with_too_few_touches():
    tracker = ZoneTracker()
    feed(tracker, [swing(10, 100.0)])
    assert tracker.nearest(price=100.0, bar_index=100, min_touches=2) is None


def test_touching_reports_every_zone_the_bar_overlapped():
    tracker = ZoneTracker(tolerance_atr=0.5)
    feed(tracker, [swing(10, 100.0), swing(20, 100.2),
                   swing(30, 102.0), swing(40, 102.1)])
    hit = tracker.touching(high=102.3, low=99.9, bar_index=100, min_touches=2)
    assert len(hit) == 2


def test_a_bar_that_misses_every_zone_reports_nothing():
    tracker = ZoneTracker()
    feed(tracker, [swing(10, 100.0), swing(20, 100.2)])
    assert tracker.touching(high=90.0, low=88.0, bar_index=100) == []


# --- 設定 -----------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_a_non_positive_tolerance_is_rejected(bad):
    with pytest.raises(ValueError):
        ZoneTracker(tolerance_atr=bad)


def test_a_swing_with_no_atr_is_skipped_rather_than_dividing_by_zero():
    tracker = ZoneTracker()
    assert tracker.update(swing(10, 100.0), atr=0.0, bar_index=13) is None
    assert tracker.zones(bar_index=100) == []
