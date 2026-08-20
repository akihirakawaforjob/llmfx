"""HistData の一括取得コマンドのテスト.

このコマンドが無かった頃、取得手順は使い捨てのスクリプトの中にしか
無かった。環境を作り直すたびにデータの作り方ごと消えていたので、
リポジトリの中へ移した。

守りたいのは 2 つ:
  1 銘柄が壊れても、残りの銘柄は最後まで取り切ること
  26 年分の M1 を抱え込まないこと(実際にメモリ不足で落ちた)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llmfx.cli import main
from llmfx.data.csv_source import load_candles_csv
from llmfx.domain.types import Candle

UTC = timezone.utc
T0 = datetime(2018, 1, 1, tzinfo=UTC)


def minutes(n: int, base: float = 100.0) -> list[Candle]:
    return [
        Candle(time=T0 + timedelta(minutes=i), open=base, high=base + 0.2,
               low=base - 0.2, close=base + 0.1, volume=1.0)
        for i in range(n)
    ]


@pytest.fixture
def fake_histdata(monkeypatch, tmp_path):
    """ダウンロードを差し替える。`broken` に入れた銘柄は壊れた zip を返す。"""
    state = {"broken": set(), "calls": [], "loaded": []}

    def download_year(symbol, year):
        path = tmp_path / f"{symbol}_{year}.zip"
        path.write_bytes(b"not-a-zip" if symbol in state["broken"] else b"ok")
        state["calls"].append((symbol, year))
        return str(path)

    def download_month(symbol, year, month):
        raise RuntimeError("この年はまだ無い")

    def load_zip(path):
        if open(path, "rb").read() == b"not-a-zip":
            raise ValueError("File is not a zip file")
        state["loaded"].append(path)
        return minutes(180)

    monkeypatch.setattr("llmfx.data.histdata_source.download_year", download_year)
    monkeypatch.setattr("llmfx.data.histdata_source.download_month", download_month)
    monkeypatch.setattr("llmfx.data.histdata_source.load_zip", load_zip)
    return state


def run(out_dir, *symbols, granularity="H1"):
    return main([
        "data", "fetch-histdata", "--symbols", *symbols,
        "--granularity", granularity, "--out-dir", str(out_dir),
        "--from-year", "2018", "--to-year", "2019",
    ])


def test_writes_one_file_per_symbol(fake_histdata, tmp_path):
    assert run(tmp_path, "USDJPY", "EURUSD") == 0
    for name in ("usdjpy_h1.csv", "eurusd_h1.csv"):
        written = load_candles_csv(tmp_path / name)
        assert written, f"{name} が空"


def test_one_broken_symbol_does_not_stop_the_others(fake_histdata, tmp_path):
    """404 の HTML が zip として届くことがある。そこで全体を止めない。"""
    fake_histdata["broken"].add("EURUSD")
    assert run(tmp_path, "USDJPY", "EURUSD", "GBPJPY") == 0
    assert (tmp_path / "usdjpy_h1.csv").exists()
    assert (tmp_path / "gbpjpy_h1.csv").exists()
    assert not (tmp_path / "eurusd_h1.csv").exists(), "壊れた銘柄を書いてはいけない"


def test_every_symbol_failing_is_reported_as_an_error(fake_histdata, tmp_path):
    fake_histdata["broken"].update({"USDJPY", "EURUSD"})
    assert run(tmp_path, "USDJPY", "EURUSD") == 2


def test_downloaded_archives_are_deleted(fake_histdata, tmp_path):
    """M1 の zip を残すと、30 銘柄 x 26 年でディスクが埋まる。"""
    assert run(tmp_path, "USDJPY") == 0
    assert not list(tmp_path.glob("*.zip")), "zip が残っている"


def test_minutes_are_aggregated_to_the_requested_granularity(fake_histdata, tmp_path):
    assert run(tmp_path, "USDJPY", granularity="H1") == 0
    h1 = load_candles_csv(tmp_path / "usdjpy_h1.csv")
    assert len(h1) == 3, "180 分は H1 で 3 本になるはず"
    assert all(c.time.minute == 0 for c in h1)


def test_skip_existing_leaves_the_file_alone(fake_histdata, tmp_path):
    target = tmp_path / "usdjpy_h1.csv"
    target.write_bytes(b"x" * 2_000_001)
    assert main([
        "data", "fetch-histdata", "--symbols", "USDJPY",
        "--out-dir", str(tmp_path), "--from-year", "2018", "--to-year", "2019",
        "--skip-existing",
    ]) == 0
    assert target.stat().st_size == 2_000_001, "既存を上書きしている"
    assert not fake_histdata["calls"], "飛ばすと言いながら取得している"
