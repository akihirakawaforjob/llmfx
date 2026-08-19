"""GMOコイン Public API アダプタのテスト.

ネットワークには出ず、httpx.MockTransport で応答を差し替える。
一番の要点は日付境界。GMO の intraday は `date` の区切りが日本時間 6:00 なので、
date=D の応答には D-1 と D の 2 日分の足が混ざる。前後 1 日を余分に取って
openTime で重複を落とす実装が正しく効いているかを確認する。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from llmfx.data.gmo import (
    EARLIEST_INTRADAY,
    GmoError,
    _date_keys,
    fetch_klines,
    to_gmo_interval,
)

UTC = timezone.utc


def ms(dt: datetime) -> str:
    return str(int(dt.timestamp() * 1000))


def bar(dt: datetime, price: float) -> dict:
    return {
        "openTime": ms(dt),
        "open": str(price), "high": str(price + 10),
        "low": str(price - 10), "close": str(price + 5), "volume": "1.5",
    }


def client_returning(handler) -> httpx.Client:
    return httpx.Client(
        base_url="https://example.invalid", transport=httpx.MockTransport(handler)
    )


def day_response(request: httpx.Request) -> httpx.Response:
    """実際の GMO と同じく、date=D は UTC で D-1 21:00 から 15 分足 96 本を返す。"""
    day = datetime.strptime(request.url.params["date"], "%Y%m%d").replace(tzinfo=UTC)
    first = day.replace(hour=21) - timedelta(days=1)
    rows = [bar(first + timedelta(minutes=15 * i), 1000 + i) for i in range(96)]
    return httpx.Response(200, json={"status": 0, "data": rows})


# --- 足の表記 -----------------------------------------------------------


def test_accepts_both_oanda_and_gmo_notation():
    assert to_gmo_interval("M15") == "15min"
    assert to_gmo_interval("15min") == "15min"
    assert to_gmo_interval("h1") == "1hour"


def test_rejects_unknown_granularity():
    with pytest.raises(GmoError, match="未対応の足"):
        to_gmo_interval("M7")


# --- date パラメータの列挙 ----------------------------------------------


def test_intraday_date_keys_pad_one_day_on_each_side():
    """JST 6:00 区切りの取りこぼしを防ぐため、要求期間の前後を 1 日ずつ広げる。"""
    keys = _date_keys(
        "15min", datetime(2024, 3, 10, tzinfo=UTC), datetime(2024, 3, 12, tzinfo=UTC)
    )
    assert keys == ["20240309", "20240310", "20240311", "20240312", "20240313"]


def test_intraday_date_keys_never_go_before_the_earliest_available_day():
    keys = _date_keys(
        "15min", datetime(2019, 1, 1, tzinfo=UTC), datetime(2021, 4, 16, tzinfo=UTC)
    )
    assert keys[0] == EARLIEST_INTRADAY.strftime("%Y%m%d")


def test_long_intervals_are_addressed_by_year():
    keys = _date_keys(
        "1day", datetime(2022, 6, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)
    )
    assert keys == ["2022", "2023", "2024"]


# --- 取得本体 -----------------------------------------------------------


def test_overlapping_days_are_deduplicated_and_clipped_to_the_range():
    """date=D の応答が前日分を含んでいても、二重取得にも欠落にもならない。"""
    with client_returning(day_response) as client:
        candles = fetch_klines(
            "BTC_JPY", "M15",
            datetime(2024, 3, 10, tzinfo=UTC), datetime(2024, 3, 11, tzinfo=UTC),
            client=client, pause=0,
        )

    times = [c.time for c in candles]
    assert len(times) == len(set(times)), "重複した足が残っている"
    assert times == sorted(times), "時系列が昇順になっていない"
    assert times[0] >= datetime(2024, 3, 10, tzinfo=UTC)
    assert times[-1] < datetime(2024, 3, 11, tzinfo=UTC)
    assert len(times) == 96, "境界で足が欠落している"


def test_days_without_data_are_treated_as_empty_not_as_failure():
    """上場前やメンテ日は ERR-5207 が返る。これで全体を落とさない。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["date"] == "20240310":
            return httpx.Response(200, json={
                "status": 2,
                "messages": [{"message_code": "ERR-5207", "message_string": "Not found"}],
            })
        return day_response(request)

    with client_returning(handler) as client:
        candles = fetch_klines(
            "BTC_JPY", "M15",
            datetime(2024, 3, 9, tzinfo=UTC), datetime(2024, 3, 12, tzinfo=UTC),
            client=client, pause=0,
        )
    assert candles, "他の日のデータまで失われている"


def test_future_dates_return_404_and_are_treated_as_empty():
    """期間の末尾は JST 6:00 区切りのため 1 日先まで問い合わせる。

    その 1 日先がまだ来ていない日だと GMO は JSON ではなく HTTP 404 を返す。
    ここを例外にすると「今日まで」の取得が必ず最後の 1 リクエストで落ちる。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["date"] >= "20260820":
            return httpx.Response(404, text="Not Found")
        return day_response(request)

    with client_returning(handler) as client:
        candles = fetch_klines(
            "BTC_JPY", "M15",
            datetime(2026, 8, 17, tzinfo=UTC), datetime(2026, 8, 19, tzinfo=UTC),
            client=client, pause=0,
        )
    assert len(candles) == 96 * 2, "末尾の 404 で取得全体が落ちている"


def test_real_http_errors_still_raise():
    """404 だけを空扱いにする。500 などは今までどおり失敗させる。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with client_returning(handler) as client:
        with pytest.raises(GmoError, match="取得に失敗"):
            fetch_klines(
                "BTC_JPY", "M15",
                datetime(2024, 3, 10, tzinfo=UTC), datetime(2024, 3, 11, tzinfo=UTC),
                client=client, pause=0, retries=0,
            )


def test_api_level_errors_are_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "status": 1,
            "messages": [{"message_code": "ERR-5003", "message_string": "Too many requests"}],
        })

    with client_returning(handler) as client:
        with pytest.raises(GmoError, match="ERR-5003"):
            fetch_klines(
                "BTC_JPY", "M15",
                datetime(2024, 3, 10, tzinfo=UTC), datetime(2024, 3, 11, tzinfo=UTC),
                client=client, pause=0,
            )


def test_naive_datetimes_are_rejected():
    with pytest.raises(GmoError, match="タイムゾーン"):
        fetch_klines("BTC_JPY", "M15", datetime(2024, 3, 10), datetime(2024, 3, 11))


def test_reversed_range_is_rejected():
    with pytest.raises(GmoError, match="逆転"):
        fetch_klines(
            "BTC_JPY", "M15",
            datetime(2024, 3, 11, tzinfo=UTC), datetime(2024, 3, 10, tzinfo=UTC),
        )
