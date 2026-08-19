"""HistData の取り込みのテスト.

Dukascopy は 1 日 1 ファイルで 5 年分が 1,900 リクエストになり、実際に
レート制限で使えなくなった。HistData は 1 年 1 ファイルなので 6 リクエスト。
実測で 1 年分 3 秒。

いちばんの罠は時刻。EST 固定(UTC-5)で夏時間の調整が無いため、素朴に
「アメリカ東部時間」として扱うと夏の半年が 1 時間ずれる。
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone

import pytest

from llmfx.data.histdata_source import HistDataError, load_zip, parse_rows

UTC = timezone.utc

SAMPLE = [
    "20240101 170000;140.843000;140.852000;140.843000;140.852000;0",
    "20240101 170100;140.852000;140.860000;140.840000;140.845000;0",
    "20240701 120000;161.200000;161.300000;161.100000;161.250000;0",
]


def test_est_is_fixed_and_does_not_follow_daylight_saving():
    """EST 固定なので、冬も夏も同じ 5 時間差。"""
    candles = parse_rows(SAMPLE)
    # 1 月 17:00 EST -> 22:00 UTC
    assert candles[0].time == datetime(2024, 1, 1, 22, 0, tzinfo=UTC)
    # 7 月 12:00 EST -> 17:00 UTC(夏時間なら 16:00 になってしまう)
    assert candles[2].time == datetime(2024, 7, 1, 17, 0, tzinfo=UTC)


def test_ohlc_is_parsed_in_order():
    c = parse_rows(SAMPLE)[0]
    assert (c.open, c.high, c.low, c.close) == (140.843, 140.852, 140.843, 140.852)
    assert c.volume == 0.0


def test_rows_are_sorted_by_time():
    shuffled = [SAMPLE[2], SAMPLE[0], SAMPLE[1]]
    times = [c.time for c in parse_rows(shuffled)]
    assert times == sorted(times)


def test_blank_lines_are_skipped():
    assert len(parse_rows(["", SAMPLE[0], "  "])) == 1


def test_short_row_is_an_error():
    with pytest.raises(HistDataError, match="列が足りません"):
        parse_rows(["20240101 170000;140.8;140.9"])


def test_bad_timestamp_is_an_error():
    with pytest.raises(HistDataError, match="解釈できません"):
        parse_rows(["2024-01-01 17:00:00;1;2;3;4;0"])


def test_zip_is_read(tmp_path):
    path = tmp_path / "DAT_ASCII_USDJPY_M1_2024.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("DAT_ASCII_USDJPY_M1_2024.csv", "\n".join(SAMPLE))
        archive.writestr("DAT_ASCII_USDJPY_M1_2024.txt", "説明")
    candles = load_zip(path)
    assert len(candles) == 3


def test_zip_without_csv_is_an_error(tmp_path):
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("readme.txt", "no data")
    with pytest.raises(HistDataError, match="CSV が入っていません"):
        load_zip(path)
