"""開発用 / 検証用の分割のテスト.

約 90 通りを試している以上、探索に使ったデータで良し悪しは決められない。
採用を決めるまで一度も見ていない期間を残すための仕組み。
口約束では守れないので、既定を安全側(dev)にしてあることも確認する。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llmfx.backtest.split import SplitError, describe, parse_boundary, split_candles
from llmfx.config import AppConfig
from llmfx.domain.types import Candle

UTC = timezone.utc


def series(start: datetime, count: int, step_days: int = 1) -> list[Candle]:
    return [
        Candle(
            time=start + timedelta(days=step_days * i),
            open=100.0, high=101.0, low=99.0, close=100.5, volume=1.0,
        )
        for i in range(count)
    ]


CANDLES = series(datetime(2024, 6, 1, tzinfo=UTC), 400)  # 2024-06-01 から 400 日


def test_boundary_accepts_a_plain_date():
    assert parse_boundary("2025-01-01") == datetime(2025, 1, 1, tzinfo=UTC)


def test_bad_boundary_is_rejected():
    with pytest.raises(SplitError, match="YYYY-MM-DD"):
        parse_boundary("2025/01/01")


def test_dev_is_everything_before_the_boundary():
    out = split_candles(CANDLES, "2025-01-01", "dev")
    assert out
    assert all(c.time < datetime(2025, 1, 1, tzinfo=UTC) for c in out)


def test_holdout_is_everything_from_the_boundary():
    out = split_candles(CANDLES, "2025-01-01", "holdout")
    assert out
    assert all(c.time >= datetime(2025, 1, 1, tzinfo=UTC) for c in out)


def test_dev_and_holdout_partition_the_data_without_overlap():
    dev = split_candles(CANDLES, "2025-01-01", "dev")
    hold = split_candles(CANDLES, "2025-01-01", "holdout")
    assert len(dev) + len(hold) == len(CANDLES)
    assert not ({c.time for c in dev} & {c.time for c in hold}), "期間が重なっている"


def test_all_returns_everything():
    assert len(split_candles(CANDLES, "2025-01-01", "all")) == len(CANDLES)


def test_no_boundary_means_no_split():
    for which in ("dev", "holdout", "all"):
        assert len(split_candles(CANDLES, None, which)) == len(CANDLES)


def test_unknown_split_is_rejected():
    with pytest.raises(SplitError, match="--split"):
        split_candles(CANDLES, "2025-01-01", "train")


def test_default_config_has_no_boundary():
    """既定では分割なし。使うときに明示的に設定する。"""
    assert AppConfig().backtest.holdout_start is None


def test_describe_always_names_the_period_being_looked_at():
    dev = split_candles(CANDLES, "2025-01-01", "dev")
    text = describe(dev, "2025-01-01", "dev")
    assert "開発用" in text and "2025-01-01" in text
    hold = split_candles(CANDLES, "2025-01-01", "holdout")
    assert "検証用" in describe(hold, "2025-01-01", "holdout")


def test_describe_handles_an_empty_slice():
    assert "該当なし" in describe([], "2025-01-01", "holdout")
