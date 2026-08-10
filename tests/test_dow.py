"""ダウ転換の検出テスト.

要件 1(転換時にエントリー)と要件 2(損切りは転換前の極値)が
実際にその通り動くことを確認する。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from llmfx.domain.dow import DowAnalyzer
from llmfx.domain.swings import SwingDetector
from llmfx.domain.types import Candle, Side, Trend

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def candles_from(closes: list[float], spread: float = 0.05) -> list[Candle]:
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


def downtrend_then_breakout() -> list[float]:
    """LH/LL を刻んで下げ、谷から一直線に直近スイング高値を上抜ける。

    谷(7.4)と上抜け(9.0)の間に中間スイングを作らないようにしてある。
    途中で新しいスイング高値ができると、そちらがブレイク対象になり
    「転換前の最安値」の起点も変わるため。
    """
    return [
        10.0, 9.0, 9.6, 8.6,   # 高値 9.6 / 安値 8.6
        9.2, 8.0,              # LH 9.2 / LL 8.0
        8.8,                   # LH 8.8  <- ここが上抜け対象
        7.4,                   # LL 7.4  <- 転換前の最安値
        8.0, 8.6, 9.0,         # 押し目を作らず 8.8 を上抜け
        9.4,
    ]


def analyze(closes: list[float]):
    analyzer = DowAnalyzer(
        detector=SwingDetector(left=1, right=1, atr_period=3, min_swing_atr=0.0),
        require_prior_trend=True,
    )
    events = []
    for candle in candles_from(closes):
        event = analyzer.update(candle)
        if event is not None:
            events.append(event)
    return analyzer, events


def test_bullish_reversal_is_detected_after_downtrend():
    analyzer, events = analyze(downtrend_then_breakout())
    bullish = [e for e in events if e.side is Side.LONG]
    assert bullish, "下降トレンド中の高値上抜けで強気転換が出ること"
    event = bullish[0]
    assert event.previous_trend is Trend.DOWN
    assert analyzer.trend is Trend.UP


def test_stop_basis_is_the_extreme_before_the_reversal():
    """要件 2: 買い転換の損切り根拠は『転換前の最安値』であること。"""
    closes = downtrend_then_breakout()
    _analyzer, events = analyze(closes)
    event = next(e for e in events if e.side is Side.LONG)

    lowest_before = min(closes[: event.bar_index + 1]) - 0.05  # 安値はスプレッド分下
    assert abs(event.stop_basis - lowest_before) < 1e-9, (
        f"損切り根拠 {event.stop_basis} は転換前の最安値 {lowest_before} と一致すべき"
    )
    assert event.stop_basis < event.candle.close


def test_no_repeat_signal_on_the_same_level():
    """同じスイング水準で何度もシグナルを出さない。"""
    closes = downtrend_then_breakout() + [9.5, 9.6, 9.7, 9.8]
    _analyzer, events = analyze(closes)
    bullish = [e for e in events if e.side is Side.LONG]
    assert len(bullish) == 1


def test_continuation_in_uptrend_is_not_a_reversal():
    """上昇トレンド中の高値更新は継続であって転換ではない。"""
    closes = [1.0, 2.0, 1.5, 3.0, 2.0, 4.0, 2.5, 5.0, 3.0, 6.0]
    analyzer, events = analyze(closes)
    assert analyzer.trend is Trend.UP
    assert not [e for e in events if e.side is Side.LONG], (
        "上昇継続でロング転換シグナルを出してはいけない"
    )


def test_require_prior_trend_blocks_range_breakouts():
    """require_prior_trend=True ならレンジ抜けでは転換扱いしない。"""
    closes = [1.0, 1.2, 1.0, 1.2, 1.0, 1.5]
    strict = DowAnalyzer(
        detector=SwingDetector(left=1, right=1, atr_period=3, min_swing_atr=0.0),
        require_prior_trend=True,
    )
    loose = DowAnalyzer(
        detector=SwingDetector(left=1, right=1, atr_period=3, min_swing_atr=0.0),
        require_prior_trend=False,
    )
    strict_events = [e for c in candles_from(closes) if (e := strict.update(c))]
    loose_events = [e for c in candles_from(closes) if (e := loose.update(c))]
    assert len(strict_events) <= len(loose_events)
