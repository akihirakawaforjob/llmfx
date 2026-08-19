"""上位足への集約のテスト.

Dukascopy は 1 分足でしか出ないため、M15 で検証するには集約が要る。
無い時間帯を埋めないこと(存在しない値動きを検証に混ぜないこと)が要点。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llmfx.data.resample import resample_candles
from llmfx.domain.types import Candle

UTC = timezone.utc
START = datetime(2024, 1, 2, tzinfo=UTC)


def m1(index: int, o: float, h: float, low: float, c: float, v: float = 1.0) -> Candle:
    return Candle(time=START + timedelta(minutes=index), open=o, high=h, low=low, close=c, volume=v)


def test_empty_input():
    assert resample_candles([], 15) == []


def test_ohlc_is_aggregated_correctly():
    bars = [
        m1(0, 100.0, 101.0, 99.5, 100.5),
        m1(1, 100.5, 103.0, 100.0, 102.0),
        m1(2, 102.0, 102.5, 98.0, 99.0),
    ]
    out = resample_candles(bars, 15)
    assert len(out) == 1
    c = out[0]
    assert c.open == 100.0          # 最初の始値
    assert c.high == 103.0          # 期間中の最高
    assert c.low == 98.0            # 期間中の最安
    assert c.close == 99.0          # 最後の終値
    assert c.volume == pytest.approx(3.0)


def test_buckets_are_anchored_to_the_epoch():
    """M15 なら 00:00, 00:15, 00:30 と綺麗に並ぶ。"""
    bars = [m1(i, 100.0, 100.0, 100.0, 100.0) for i in range(45)]
    out = resample_candles(bars, 15)
    assert [c.time.minute for c in out] == [0, 15, 30]


def test_gaps_do_not_create_empty_buckets():
    """FX の週末など足の無い区間は、その区切り自体を作らない。"""
    bars = [m1(0, 100.0, 100.0, 100.0, 100.0), m1(120, 200.0, 200.0, 200.0, 200.0)]
    out = resample_candles(bars, 15)
    assert len(out) == 2, "存在しない時間帯を埋めている"
    assert out[0].time == START
    assert out[1].time == START + timedelta(minutes=120)


def test_partial_final_bucket_is_kept():
    bars = [m1(i, 100.0, 100.0, 100.0, 100.0) for i in range(20)]
    out = resample_candles(bars, 15)
    assert len(out) == 2
    assert out[1].time == START + timedelta(minutes=15)


def test_invalid_interval_is_rejected():
    with pytest.raises(ValueError):
        resample_candles([m1(0, 1.0, 1.0, 1.0, 1.0)], 0)
