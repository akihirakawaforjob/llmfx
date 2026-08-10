"""CSV 形式のローソク足の読み書き.

列: time,open,high,low,close,volume
time は ISO8601(UTC)。
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ..domain.types import Candle

FIELDNAMES = ["time", "open", "high", "low", "close", "volume"]


def _parse_time(value: str) -> datetime:
    text = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_candles_csv(path: str | Path) -> list[Candle]:
    candles: list[Candle] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDNAMES[:5]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV に必要な列がありません: {sorted(missing)}")
        for row in reader:
            candles.append(
                Candle(
                    time=_parse_time(row["time"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume") or 0.0),
                )
            )
    candles.sort(key=lambda c: c.time)
    return candles


def save_candles_csv(candles: Iterable[Candle], path: str | Path) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for candle in candles:
            writer.writerow(
                {
                    "time": candle.time.astimezone(timezone.utc).isoformat(),
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
            )
            count += 1
    return count
