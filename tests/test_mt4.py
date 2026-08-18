"""MT4 / MT5 の CSV 取り込みのテスト.

エクスポート形式はプラットフォームと設定で変わるため、実際に出てくる
主要な 3 形式を通す。時刻の変換を間違えると足の並びが崩れるので、
タイムゾーン補正も明示的に確認する。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llmfx.data.mt4 import Mt4FormatError, infer_granularity_minutes, load_mt4_csv

# MT4 ヒストリーセンター: ヘッダなし・カンマ区切り・日付と時刻が別列
MT4_PLAIN = """\
2024.01.02,00:00,140.123,140.200,140.100,140.150,1234
2024.01.02,00:15,140.150,140.260,140.140,140.240,987
2024.01.02,00:30,140.240,140.300,140.180,140.190,1500
"""

# MT5: ヘッダあり・タブ区切り・末尾に余分な列
MT5_TABBED = """\
<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
2024.01.02\t00:00:00\t140.123\t140.200\t140.100\t140.150\t1234\t0\t12
2024.01.02\t00:15:00\t140.150\t140.260\t140.140\t140.240\t987\t0\t11
"""

# 日時が 1 列にまとまっている形式
SINGLE_COLUMN = """\
2024-01-02 00:00,140.123,140.200,140.100,140.150,1234
2024-01-02 00:15,140.150,140.260,140.140,140.240,987
"""


def write(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_reads_plain_mt4_export(tmp_path):
    candles = load_mt4_csv(write(tmp_path, "m.csv", MT4_PLAIN))
    assert len(candles) == 3
    first = candles[0]
    assert first.time == datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    assert first.open == pytest.approx(140.123)
    assert first.high == pytest.approx(140.200)
    assert first.low == pytest.approx(140.100)
    assert first.close == pytest.approx(140.150)
    assert first.volume == pytest.approx(1234)


def test_reads_tab_separated_mt5_export_with_header(tmp_path):
    candles = load_mt4_csv(write(tmp_path, "m5.csv", MT5_TABBED))
    assert len(candles) == 2
    assert candles[0].time == datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
    assert candles[1].close == pytest.approx(140.240)


def test_reads_single_datetime_column(tmp_path):
    candles = load_mt4_csv(write(tmp_path, "s.csv", SINGLE_COLUMN))
    assert len(candles) == 2
    assert candles[0].time == datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)


def test_server_timezone_offset_shifts_to_utc(tmp_path):
    """MT4 サーバが GMT+2 なら、2 時間引いて UTC になること。"""
    path = write(tmp_path, "m.csv", MT4_PLAIN)
    utc = load_mt4_csv(path, server_tz_offset=0.0)
    shifted = load_mt4_csv(path, server_tz_offset=2.0)
    assert (utc[0].time - shifted[0].time).total_seconds() == 2 * 3600


def test_candles_are_sorted_even_if_the_file_is_not(tmp_path):
    reversed_body = "\n".join(reversed(MT4_PLAIN.strip().splitlines())) + "\n"
    candles = load_mt4_csv(write(tmp_path, "r.csv", reversed_body))
    assert [c.time for c in candles] == sorted(c.time for c in candles)


def test_granularity_is_inferred(tmp_path):
    candles = load_mt4_csv(write(tmp_path, "m.csv", MT4_PLAIN))
    assert infer_granularity_minutes(candles) == pytest.approx(15.0)


def test_empty_file_is_rejected(tmp_path):
    with pytest.raises(Mt4FormatError, match="空"):
        load_mt4_csv(write(tmp_path, "e.csv", "\n\n"))


def test_header_only_file_is_rejected(tmp_path):
    with pytest.raises(Mt4FormatError, match="1 本も"):
        load_mt4_csv(write(tmp_path, "h.csv", "<DATE>\t<TIME>\t<OPEN>\n"))


def test_truncated_row_reports_the_line_number(tmp_path):
    body = MT4_PLAIN + "2024.01.02,00:45,140.190\n"
    with pytest.raises(Mt4FormatError, match="4 行目"):
        load_mt4_csv(write(tmp_path, "t.csv", body))


def test_imported_data_survives_a_round_trip_through_our_csv(tmp_path):
    """取り込んだ足が、そのままバックテストの入力形式で読み書きできること。"""
    from llmfx.data.csv_source import load_candles_csv, save_candles_csv

    candles = load_mt4_csv(write(tmp_path, "m.csv", MT4_PLAIN))
    out = tmp_path / "converted.csv"
    save_candles_csv(candles, out)
    restored = load_candles_csv(out)

    assert len(restored) == len(candles)
    assert restored[0].time == candles[0].time
    assert restored[0].close == pytest.approx(candles[0].close)
