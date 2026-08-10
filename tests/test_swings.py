"""スイング検出のテスト.

最重要の性質は「確定が遅れること」。ピボットは右 N 本を見ないと確定
できないため、確定前に参照できてしまうと先読みになる。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from llmfx.domain.swings import SwingDetector
from llmfx.domain.types import Candle, SwingLabel, SwingType

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def make_candles(closes: list[float], spread: float = 0.05) -> list[Candle]:
    return [
        Candle(
            time=START + timedelta(minutes=15 * i),
            open=c,
            high=c + spread,
            low=c - spread,
            close=c,
        )
        for i, c in enumerate(closes)
    ]


def test_pivot_high_confirms_only_after_right_bars():
    # 山の頂点は index 5。left=right=2 なので確定は index 7。
    closes = [1.0, 1.1, 1.2, 1.3, 1.4, 2.0, 1.4, 1.3, 1.2, 1.1]
    detector = SwingDetector(left=2, right=2, atr_period=2, min_swing_atr=0.0)

    confirmed_at = None
    for i, candle in enumerate(make_candles(closes)):
        detector.update(candle)
        if detector.last_swing(SwingType.HIGH) is not None and confirmed_at is None:
            confirmed_at = i

    assert confirmed_at == 7, "右 2 本を見終わるまでスイング高値は確定してはいけない"
    swing = detector.last_swing(SwingType.HIGH)
    assert swing is not None
    assert swing.index == 5
    assert swing.confirmed_index == 7


def test_swings_alternate_between_high_and_low():
    closes = [1.0, 2.0, 1.0, 3.0, 1.5, 4.0, 2.0, 5.0, 2.5, 6.0, 3.0, 7.0, 3.5]
    detector = SwingDetector(left=1, right=1, atr_period=2, min_swing_atr=0.0)
    for candle in make_candles(closes):
        detector.update(candle)

    types = [s.type for s in detector.swings]
    assert len(types) >= 4
    for previous, current in zip(types, types[1:]):
        assert previous is not current, "高値と安値は必ず交互に並ぶこと"


def test_noise_below_threshold_is_filtered():
    """min_swing_atr を超えない揺れはスイングとして採用しない。"""
    closes = [1.0, 1.001, 1.0, 1.001, 1.0, 1.001, 1.0]
    detector = SwingDetector(left=1, right=1, atr_period=2, min_swing_atr=5.0)
    for candle in make_candles(closes, spread=0.0005):
        detector.update(candle)
    assert len(detector.swings) <= 1


def test_higher_high_and_higher_low_are_labelled():
    closes = [1.0, 2.0, 1.5, 3.0, 2.0, 4.0, 2.5]
    detector = SwingDetector(left=1, right=1, atr_period=2, min_swing_atr=0.0)
    for candle in make_candles(closes):
        detector.update(candle)

    highs = detector.swings_of(SwingType.HIGH)
    lows = detector.swings_of(SwingType.LOW)
    assert len(highs) >= 2 and len(lows) >= 2
    assert highs[-1].label is SwingLabel.HH
    assert lows[-1].label is SwingLabel.HL


def test_consecutive_same_type_keeps_the_more_extreme():
    """同じ向きが続いたら、より極端な方だけが残る(ジグザグ化)。"""
    detector = SwingDetector(left=1, right=1, atr_period=2, min_swing_atr=0.0)
    for candle in make_candles([1.0, 2.0, 1.9, 3.0, 1.0]):
        detector.update(candle)

    highs = detector.swings_of(SwingType.HIGH)
    assert highs, "高値が検出されていること"
    assert highs[-1].price >= 3.0 - 0.06
