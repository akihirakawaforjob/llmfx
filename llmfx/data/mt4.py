"""MT4 / MT5 がエクスポートした CSV の読み込み.

楽天 MT4 などのプラットフォームから出した履歴データを、本システムの
ローソク足へ変換する。エクスポート形式は環境によって細部が違うため、
区切り文字・ヘッダ有無・列構成を自動判定する。

対応する形式:

    MT4 ヒストリーセンター(ヘッダなし・カンマ区切り)
        2024.01.02,00:00,140.123,140.200,140.100,140.150,1234

    MT5 (ヘッダあり・タブ区切り、末尾に余分な列)
        <DATE>	<TIME>	<OPEN>	<HIGH>	<LOW>	<CLOSE>	<TICKVOL>	<VOL>	<SPREAD>
        2024.01.02	00:00:00	140.123	140.200	140.100	140.150	1234	0	12

    日時が 1 列になっている形式
        2024-01-02 00:00,140.123,140.200,140.100,140.150,1234

時刻の扱いが最重要:
MT4 のサーバ時刻は UTC ではなく、業者ごとに GMT+2 / GMT+3 などへずれている。
そのままだと日足の区切りも取引時間帯もずれるため、`server_tz_offset` で
明示的に UTC へ直す。分からない場合は 0 のままでもバックテストの
売買ロジックには影響しないが、時間帯フィルタを足すときに効いてくる。
"""

from __future__ import annotations

import csv
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..domain.types import Candle

_DATE = re.compile(r"^\d{4}[./-]\d{1,2}[./-]\d{1,2}$")
_TIME = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
_DATETIME = re.compile(r"^\d{4}[./-]\d{1,2}[./-]\d{1,2}[ T]\d{1,2}:\d{2}(:\d{2})?$")


class Mt4FormatError(ValueError):
    pass


def _sniff_delimiter(sample: str) -> str:
    """行に最も多く現れる区切り文字を採用する。"""
    counts = {d: sample.count(d) for d in (",", "\t", ";")}
    delimiter = max(counts, key=counts.get)
    if counts[delimiter] == 0:
        raise Mt4FormatError(
            "区切り文字(カンマ / タブ / セミコロン)が見つかりません。"
            "MT4 のヒストリーセンターからエクスポートした CSV か確認してください。"
        )
    return delimiter


def _parse_timestamp(fields: list[str]) -> tuple[datetime, int]:
    """先頭の日時を読み、(日時, 消費した列数) を返す。"""
    first = fields[0].strip()

    if _DATE.match(first) and len(fields) > 1 and _TIME.match(fields[1].strip()):
        stamp = f"{first} {fields[1].strip()}"
        consumed = 2
    elif _DATETIME.match(first):
        stamp = first
        consumed = 1
    else:
        raise Mt4FormatError(f"日時として解釈できません: {first!r}")

    normalized = stamp.replace(".", "-").replace("/", "-").replace("T", " ")
    date_part, time_part = normalized.split(" ", 1)
    if time_part.count(":") == 1:
        time_part += ":00"
    year, month, day = (int(v) for v in date_part.split("-"))
    hour, minute, second = (int(v) for v in time_part.split(":"))
    return datetime(year, month, day, hour, minute, second), consumed


def load_mt4_csv(
    path: str | Path,
    server_tz_offset: float = 0.0,
) -> list[Candle]:
    """MT4 / MT5 形式の CSV を読み込む。

    `server_tz_offset` はデータの時刻が UTC から何時間ずれているか。
    MT4 のサーバ時刻が GMT+2 なら 2 を渡す(= 2 時間引いて UTC にする)。
    """
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise Mt4FormatError("ファイルが空です")

    delimiter = _sniff_delimiter(lines[0] if len(lines) < 2 else lines[1])
    shift = timedelta(hours=server_tz_offset)

    candles: list[Candle] = []
    skipped_header = 0
    for line_no, row in enumerate(csv.reader(lines, delimiter=delimiter), start=1):
        fields = [f.strip() for f in row if f.strip() != ""]
        if not fields:
            continue
        # <DATE> のようなヘッダ行、および数値でない先頭行は読み飛ばす。
        if fields[0].startswith("<") or not fields[0][:1].isdigit():
            skipped_header += 1
            continue

        try:
            stamp, consumed = _parse_timestamp(fields)
            values = fields[consumed:]
            if len(values) < 4:
                raise Mt4FormatError(f"OHLC の列が足りません(列数 {len(fields)})")
            open_, high, low, close = (float(v) for v in values[:4])
            volume = float(values[4]) if len(values) > 4 else 0.0
        except (ValueError, Mt4FormatError) as exc:
            raise Mt4FormatError(f"{line_no} 行目を解釈できません: {exc}\n  該当行: {line_no}") from exc

        candles.append(
            Candle(
                time=(stamp - shift).replace(tzinfo=timezone.utc),
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )

    if not candles:
        raise Mt4FormatError(
            "ローソク足を 1 本も読み取れませんでした。"
            f"(ヘッダとして読み飛ばした行: {skipped_header})"
        )

    candles.sort(key=lambda c: c.time)
    return candles


def infer_granularity_minutes(candles: list[Candle]) -> float | None:
    """足の間隔を推定する(取り込み結果の確認用)。"""
    if len(candles) < 3:
        return None
    gaps = [
        (b.time - a.time).total_seconds() / 60.0
        for a, b in zip(candles, candles[1:])
    ]
    gaps.sort()
    return gaps[len(gaps) // 2]  # 週末の空白に引きずられないよう中央値
