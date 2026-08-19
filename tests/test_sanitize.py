"""壊れた足の除去のテスト.

HistData の初期には実在しない値が混ざっている。1 本混ざるだけで
バックテストが壊れる(実測で 1 件の取引が -277 R を計上した)。

同時に、**本物の急変を落としてはいけない**。2015-01-15 のスイスフラン
ショックでは USD/CHF が数分で 18% 動いている。閾値をきつくすると、
最も重要な場面を検証から外すことになる。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from llmfx.data.sanitize import drop_bad_bars
from llmfx.domain.types import Candle

UTC = timezone.utc
T0 = datetime(2001, 6, 8, tzinfo=UTC)


def bar(i: int, price: float, high: float | None = None, low: float | None = None) -> Candle:
    return Candle(
        time=T0 + timedelta(minutes=5 * i),
        open=price,
        high=high if high is not None else price * 1.001,
        low=low if low is not None else price * 0.999,
        close=price,
        volume=1.0,
    )


def normal(count: int, price: float = 0.8485) -> list[Candle]:
    return [bar(i, price + 0.0001 * (i % 5)) for i in range(count)]


def test_clean_data_passes_through():
    candles = normal(300)
    kept, report = drop_bad_bars(candles)
    assert len(kept) == len(candles)
    assert report.clean


def test_absurd_spike_is_dropped():
    """EURUSD 2001-06-08 の 0.84850 → 1965.00010。"""
    candles = normal(250)
    candles.insert(200, bar(200, 1965.0001))
    kept, report = drop_bad_bars(candles)
    assert report.dropped == 1
    assert all(c.close < 2.0 for c in kept)


def test_negative_price_is_dropped():
    """EURUSD 2001-09-11 の -0.00010。"""
    candles = normal(250)
    candles.insert(200, Candle(time=T0, open=-0.0001, high=-0.0001, low=-0.0001,
                               close=-0.0001, volume=0.0))
    kept, report = drop_bad_bars(candles)
    assert report.dropped == 1
    assert all(c.close > 0 for c in kept)


def test_collapsed_price_is_dropped():
    """AUDUSD 2000-08-04 の 0.57950 → 0.10060。"""
    candles = normal(250, price=0.5795)
    candles.insert(200, bar(200, 0.1006))
    kept, report = drop_bad_bars(candles)
    assert report.dropped == 1


def test_malformed_ohlc_is_dropped():
    candles = normal(250)
    candles.insert(200, bar(200, 0.85, high=0.80, low=0.90))  # 高安が逆
    kept, report = drop_bad_bars(candles)
    assert report.dropped == 1
    assert "形が壊れている" in report.reasons


def test_a_real_18_percent_move_survives():
    """スイスフランショックを落とさないこと。これを落としたら意味が無い。"""
    candles = normal(300, price=1.02)
    shocked = [bar(300 + i, 0.84) for i in range(30)]  # 18% 下落して定着
    kept, report = drop_bad_bars(candles + shocked)
    assert report.dropped == 0, f"本物の急変を落としている: {report.reasons}"
    assert len(kept) == 330


def test_a_sustained_trend_is_not_dropped():
    """何ヶ月もかけて 2 倍になるような動きは正常。中央値が追従する。"""
    candles = [bar(i, 100.0 * (1.0 + 0.004 * i)) for i in range(400)]  # 最終的に 2.6 倍
    kept, report = drop_bad_bars(candles)
    assert report.dropped == 0, f"緩やかなトレンドを落としている: {report.reasons}"


def test_the_outlier_itself_does_not_poison_the_reference():
    """異常値を基準に取り込むと、その後の正常な足まで落ち始める。"""
    candles = normal(250)
    for offset in (200, 201, 202):
        candles.insert(offset, bar(offset, 1965.0001))
    kept, report = drop_bad_bars(candles)
    assert report.dropped == 3
    assert len(kept) == 250


def test_report_reads_clearly():
    candles = normal(250)
    candles.insert(200, bar(200, 1965.0001))
    _, report = drop_bad_bars(candles)
    assert "除去 1 本" in report.summary()
    assert report.examples
