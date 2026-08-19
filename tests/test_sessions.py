"""時間帯・イベントフィルタのテスト.

要点は先読みを持ち込まないこと。ここで使ってよいのは、その時点で
手に入る情報だけ:
  - 時刻・曜日          カレンダーから事前に分かる
  - 主要指標の予定      雇用統計は毎月第 1 金曜。事前に分かる
  - 直近のボラティリティ 過去のバーだけから計算できる
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llmfx.config import AppConfig, ConfigError
from llmfx.data.synthetic import generate_synthetic_candles
from llmfx.domain.sessions import SessionFilter, nfp_time
from llmfx.domain.strategy import DowReversalStrategy

UTC = timezone.utc


def at(y: int, m: int, d: int, h: int = 12, mi: int = 0) -> datetime:
    return datetime(y, m, d, h, mi, tzinfo=UTC)


# --- 素通し ---------------------------------------------------------------


def test_default_filter_lets_everything_through():
    allowed, reason = SessionFilter().allows(at(2024, 3, 13, 3))
    assert allowed and reason is None


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError, match="タイムゾーン"):
        SessionFilter().allows(datetime(2024, 3, 13, 12))


# --- 時間帯 ---------------------------------------------------------------


def test_hours_outside_the_allowed_window_are_blocked():
    f = SessionFilter(allowed_hours_utc=((7, 16),))
    assert f.allows(at(2024, 3, 13, 7))[0]
    assert f.allows(at(2024, 3, 13, 15))[0]
    assert not f.allows(at(2024, 3, 13, 6))[0]
    assert not f.allows(at(2024, 3, 13, 16))[0], "終了時は含めない"
    assert f.allows(at(2024, 3, 13, 3))[1] == "outside_session"


def test_window_can_wrap_around_midnight():
    """東京時間のように日をまたぐ帯も指定できる。"""
    f = SessionFilter(allowed_hours_utc=((22, 6),))
    assert f.allows(at(2024, 3, 13, 23))[0]
    assert f.allows(at(2024, 3, 13, 2))[0]
    assert not f.allows(at(2024, 3, 13, 12))[0]


def test_multiple_windows():
    f = SessionFilter(allowed_hours_utc=((0, 2), (12, 14)))
    assert f.allows(at(2024, 3, 13, 1))[0]
    assert f.allows(at(2024, 3, 13, 13))[0]
    assert not f.allows(at(2024, 3, 13, 8))[0]


def test_empty_window_blocks_nothing_extra():
    assert SessionFilter(allowed_hours_utc=()).allows(at(2024, 3, 13, 3))[0]


# --- 曜日 -----------------------------------------------------------------


def test_blocked_weekdays():
    f = SessionFilter(blocked_weekdays=(0, 4))  # 月・金
    assert not f.allows(at(2024, 3, 11))[0]     # 月
    assert f.allows(at(2024, 3, 13))[0]         # 水
    assert not f.allows(at(2024, 3, 15))[0]     # 金
    assert f.allows(at(2024, 3, 11))[1] == "weekday_blocked"


# --- 雇用統計(事前に分かる予定) -------------------------------------------


def test_nfp_is_the_first_friday():
    # 2024-03-01 は金曜。2024-04-05 も金曜。
    assert nfp_time(2024, 3).date() == datetime(2024, 3, 1).date()
    assert nfp_time(2024, 4).date() == datetime(2024, 4, 5).date()
    assert nfp_time(2024, 11).date() == datetime(2024, 11, 1).date()


def test_nfp_hour_follows_us_daylight_saving():
    """8:30 ET は夏時間 12:30 UTC、冬時間 13:30 UTC。"""
    assert nfp_time(2024, 7).hour == 12   # 夏時間
    assert nfp_time(2024, 1).hour == 13   # 冬時間


def test_nfp_blackout_blocks_the_window_around_the_release():
    f = SessionFilter(nfp_blackout_minutes=60)
    release = nfp_time(2024, 3)
    assert not f.allows(release)[0]
    assert not f.allows(release - timedelta(minutes=59))[0]
    assert not f.allows(release + timedelta(minutes=59))[0]
    assert f.allows(release + timedelta(minutes=61))[0]
    assert f.allows(release)[1] == "nfp_blackout"


def test_nfp_blackout_spans_month_boundaries():
    """月初の発表は、前月末の時刻から見ても遮断されている必要がある。"""
    release = nfp_time(2024, 3)  # 3/1 12:30
    f = SessionFilter(nfp_blackout_minutes=24 * 60)
    assert not f.allows(release - timedelta(hours=20))[0], "前月末側が抜けている"


def test_zero_blackout_disables_the_check():
    assert SessionFilter(nfp_blackout_minutes=0).allows(nfp_time(2024, 3))[0]


# --- 設定の検証 -----------------------------------------------------------


def test_invalid_hour_span_is_rejected():
    config = AppConfig()
    config.entry.allowed_hours_utc = [[7, 30]]
    with pytest.raises(ConfigError, match="allowed_hours_utc"):
        config.validate()


def test_invalid_weekday_is_rejected():
    config = AppConfig()
    config.entry.blocked_weekdays = [7]
    with pytest.raises(ConfigError, match="blocked_weekdays"):
        config.validate()


def test_invalid_percentile_is_rejected():
    config = AppConfig()
    config.entry.max_atr_percentile = 1.5
    with pytest.raises(ConfigError, match="max_atr_percentile"):
        config.validate()


# --- 戦略への組み込み -----------------------------------------------------


def test_session_filter_reduces_signals_and_records_the_reason():
    candles = generate_synthetic_candles(count=8000, seed=20260810)

    base = AppConfig()
    base.entry.min_rr = 1e-6
    plain = DowReversalStrategy(base)
    all_signals = [s for c in candles if (s := plain.update(c))]

    narrow = AppConfig.from_dict(base.to_dict())
    narrow.entry.allowed_hours_utc = [[7, 16]]
    narrow.validate()
    filtered = DowReversalStrategy(narrow)
    kept = [s for c in candles if (s := filtered.update(c))]

    assert len(kept) < len(all_signals)
    assert all(7 <= s.time.hour < 16 for s in kept)
    assert any(r.reason == "outside_session" for r in filtered.rejections)


def test_volatility_filter_uses_only_past_bars():
    """現在の ATR を自分自身の履歴に混ぜると、比較が緩んで条件が働かない。"""
    candles = generate_synthetic_candles(count=8000, seed=20260810)

    config = AppConfig()
    config.entry.min_rr = 1e-6
    config.entry.max_atr_percentile = 0.5
    config.validate()
    strategy = DowReversalStrategy(config)
    kept = [s for c in candles if (s := strategy.update(c))]

    assert any(r.reason == "volatility_too_high" for r in strategy.rejections)
    assert kept, "すべて落ちてしまっている"


def test_volatility_filter_blocks_while_history_is_thin():
    """履歴が薄いうちは判定できない。素通しではなく見送る側へ倒す。"""
    config = AppConfig()
    config.entry.max_atr_percentile = 0.9
    config.validate()
    strategy = DowReversalStrategy(config)
    assert strategy._atr_is_calm(1.0) is False
