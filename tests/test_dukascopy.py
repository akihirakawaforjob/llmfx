"""Dukascopy アダプタのテスト.

ネットワークには出ず、実データと同じ形の bi5 を組み立てて食わせる。
要点は 3 つ:
  - URL の月が 0 始まり(1 月 = 00)。ここを間違えると 1 か月ずれる
  - レコードは (時刻秒, 始値, 終値, 安値, 高値, 出来高)。高安が後ろにある
  - 土日は 404、503 は断続的に混ざる。前者は空、後者はリトライ
"""

from __future__ import annotations

import lzma
import struct
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from llmfx.data.dukascopy import (
    DukascopyError,
    _url,
    decode_bi5,
    fetch_m1_candles,
    point_scale,
)

UTC = timezone.utc
RECORD = struct.Struct(">iiiiif")


def build_bi5(rows: list[tuple[int, int, int, int, int, float]]) -> bytes:
    """(秒, 始値, 終値, 安値, 高値, 出来高) の並びを実データと同じ形へ。"""
    raw = b"".join(RECORD.pack(*r) for r in rows)
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_ALONE)
    return compressor.compress(raw) + compressor.flush()


def minute_rows(count: int, base: int = 141_000) -> list:
    return [
        (60 * i, base + i, base + i + 8, base + i - 4, base + i + 12, 100.0 + i)
        for i in range(count)
    ]


# --- 価格スケール --------------------------------------------------------


def test_jpy_pairs_use_three_decimals():
    assert point_scale("USDJPY") == 1000
    assert point_scale("eurjpy") == 1000


def test_other_pairs_use_five_decimals():
    assert point_scale("EURUSD") == 100_000
    assert point_scale("GBPCHF") == 100_000


# --- URL(月が 0 始まり) --------------------------------------------------


def test_month_is_zero_indexed_in_the_url():
    """1 月が 00。ここを間違えると 1 か月ずれたデータを黙って掴む。"""
    url = _url("USDJPY", datetime(2024, 1, 2, tzinfo=UTC), "BID")
    assert "/2024/00/02/" in url
    assert url.endswith("BID_candles_min_1.bi5")


def test_december_is_eleven():
    url = _url("USDJPY", datetime(2024, 12, 31, tzinfo=UTC), "BID")
    assert "/2024/11/31/" in url


# --- レコードの復号 ------------------------------------------------------


def test_records_decode_with_high_and_low_after_close():
    day = datetime(2024, 1, 2, tzinfo=UTC)
    payload = build_bi5([(0, 141_110, 141_118, 141_096, 141_130, 407.03)])
    candles = decode_bi5(payload, day, scale=1000)

    assert len(candles) == 1
    c = candles[0]
    assert c.time == day
    assert c.open == pytest.approx(141.110)
    assert c.close == pytest.approx(141.118)
    assert c.low == pytest.approx(141.096)
    assert c.high == pytest.approx(141.130)
    assert c.volume == pytest.approx(407.03)
    assert c.low <= min(c.open, c.close) and c.high >= max(c.open, c.close)


def test_time_field_is_seconds_from_the_start_of_the_day():
    day = datetime(2024, 1, 2, tzinfo=UTC)
    payload = build_bi5(minute_rows(3))
    candles = decode_bi5(payload, day, scale=1000)
    assert [c.time for c in candles] == [
        day,
        day + timedelta(minutes=1),
        day + timedelta(minutes=2),
    ]


def test_empty_minutes_are_skipped():
    """取引の無かった分は価格 0 で詰まっている。足として扱わない。"""
    day = datetime(2024, 1, 2, tzinfo=UTC)
    payload = build_bi5([(0, 141_110, 141_118, 141_096, 141_130, 1.0),
                         (60, 0, 0, 0, 0, 0.0),
                         (120, 141_200, 141_210, 141_190, 141_220, 2.0)])
    candles = decode_bi5(payload, day, scale=1000)
    assert len(candles) == 2
    assert [c.time.minute for c in candles] == [0, 2]


def test_empty_payload_is_no_candles():
    assert decode_bi5(b"", datetime(2024, 1, 2, tzinfo=UTC), 1000) == []


def test_truncated_record_is_an_error():
    day = datetime(2024, 1, 2, tzinfo=UTC)
    compressor = lzma.LZMACompressor(format=lzma.FORMAT_ALONE)
    broken = compressor.compress(b"\x00" * 10) + compressor.flush()
    with pytest.raises(DukascopyError, match="レコード長"):
        decode_bi5(broken, day, 1000)


# --- 取得 ----------------------------------------------------------------


def client_returning(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


def test_missing_days_are_skipped_not_failed():
    """土日・祝日は 404。エラーではなく「データ無し」。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if "/06/" in str(request.url):  # 7 月 6 日(土)相当を欠かす
            return httpx.Response(404)
        return httpx.Response(200, content=build_bi5(minute_rows(2)))

    with client_returning(handler) as client:
        candles = fetch_m1_candles(
            "USDJPY",
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 5, tzinfo=UTC),
            client=client, pause=0,
        )
    assert candles, "404 の日で全体が落ちている"


def test_transient_503_is_retried():
    """503 が断続的に混ざる。ここを諦めると数年分の取得が途中で死ぬ。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="Service Unavailable")
        return httpx.Response(200, content=build_bi5(minute_rows(2)))

    with client_returning(handler) as client:
        candles = fetch_m1_candles(
            "USDJPY",
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 3, tzinfo=UTC),
            client=client, pause=0, backoff=0.0,
        )
    assert len(candles) == 2
    assert calls["n"] == 2, "リトライしていない"


def test_persistent_503_eventually_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    with client_returning(handler) as client:
        with pytest.raises(DukascopyError, match="503"):
            fetch_m1_candles(
                "USDJPY",
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
                client=client, pause=0, retries=2, backoff=0.0,
            )


def test_results_are_clipped_to_the_requested_range():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=build_bi5(minute_rows(1440)))

    start = datetime(2024, 1, 2, 10, 0, tzinfo=UTC)
    end = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
    with client_returning(handler) as client:
        candles = fetch_m1_candles("USDJPY", start, end, client=client, pause=0)
    assert all(start <= c.time < end for c in candles)
    assert len(candles) == 120


def test_naive_datetimes_are_rejected():
    with pytest.raises(DukascopyError, match="タイムゾーン"):
        fetch_m1_candles("USDJPY", datetime(2024, 1, 2), datetime(2024, 1, 3))


def test_reversed_range_is_rejected():
    with pytest.raises(DukascopyError, match="逆転"):
        fetch_m1_candles(
            "USDJPY",
            datetime(2024, 1, 3, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        )


def test_non_strict_mode_records_failed_days_instead_of_aborting():
    """1 日の 503 で数年分の取得が丸ごと落ちるのを避ける。

    ただし飛ばした日は failures に必ず残す。穴の空いたデータで
    バックテストを回すと、理由の分からない成績差として現れるため。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if "/03/" in str(request.url):  # 1 月 3 日だけ落ち続ける
            return httpx.Response(503, text="Service Unavailable")
        return httpx.Response(200, content=build_bi5(minute_rows(2)))

    failures: list[datetime] = []
    with client_returning(handler) as client:
        candles = fetch_m1_candles(
            "USDJPY",
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 5, tzinfo=UTC),
            client=client, pause=0, retries=1, backoff=0.0,
            strict=False, failures=failures,
        )

    assert candles, "他の日まで失われている"
    assert [d.day for d in failures] == [3], "飛ばした日が記録されていない"


def test_backoff_is_capped():
    """指数バックオフを青天井にすると 1 日分で何分も待つことになる。"""
    from llmfx.data.dukascopy import MAX_BACKOFF

    assert MAX_BACKOFF <= 60.0
