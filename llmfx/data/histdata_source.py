"""HistData.com の 1 分足を取り込む.

Dukascopy は 1 日 1 ファイルなので 5 年分で 1,900 リクエストになり、
実際にレート制限で 503 が返るようになって使えなくなった。HistData は
**1 年を 1 ファイル**で配るので、同じ 5 年分が 6 リクエストで済む。
実測で 1 年分 3 秒。桁違いに速く、負荷も掛けない。

トークン付きの POST が必要で、その流れは `histdata` パッケージが実装して
いるのでそれに任せる(requirements.txt に追加済み)。ここで受け持つのは
展開と時刻の変換。

ファイルの形式(GENERIC_ASCII / M1):
    YYYYMMDD HHMMSS;始値;高値;安値;終値;出来高
    20240101 170000;140.843000;140.852000;140.843000;140.852000;0

    セミコロン区切り。出来高は FX では常に 0。

時刻がいちばんの罠:
    **EST 固定(UTC-5)で、夏時間の調整が無い。** 「EST/EDT」ではなく
    「EST のまま一年中」なので、素朴に「アメリカ東部時間」として扱うと
    夏の半年が 1 時間ずれる。UTC へ直すには常に 5 時間足せばよい。
"""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..domain.types import Candle

# HistData の GENERIC_ASCII は EST 固定。夏時間の調整が無いので常にこの差。
EST_OFFSET_HOURS = -5

# 年単位で配られるのは前年まで。当年は月単位でしか取れない。
FIRST_YEAR = 2000


class HistDataError(RuntimeError):
    pass


def parse_rows(lines, offset_hours: int = EST_OFFSET_HOURS) -> list[Candle]:
    """`YYYYMMDD HHMMSS;o;h;l;c;v` の並びをローソク足へ。"""
    shift = timedelta(hours=offset_hours)
    candles: list[Candle] = []
    for line_no, row in enumerate(csv.reader(lines, delimiter=";"), start=1):
        if not row or not row[0].strip():
            continue
        if len(row) < 5:
            raise HistDataError(f"{line_no} 行目: 列が足りません({len(row)} 列)")
        stamp = row[0].strip()
        try:
            moment = datetime.strptime(stamp, "%Y%m%d %H%M%S")
            open_, high, low, close = (float(v) for v in row[1:5])
        except ValueError as exc:
            raise HistDataError(f"{line_no} 行目を解釈できません: {stamp!r}: {exc}") from exc
        candles.append(
            Candle(
                # EST -> UTC。offset は負なので引くと足す向きになる。
                time=(moment - shift).replace(tzinfo=timezone.utc),
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=float(row[5]) if len(row) > 5 and row[5].strip() else 0.0,
            )
        )
    candles.sort(key=lambda c: c.time)
    return candles


def load_zip(path: str | Path, offset_hours: int = EST_OFFSET_HOURS) -> list[Candle]:
    """HistData が配る ZIP をそのまま読む。"""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise HistDataError(f"ZIP に CSV が入っていません: {path}")
        with archive.open(names[0]) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
            return parse_rows(text, offset_hours)


def download_year(
    pair: str, year: int, out_dir: str | Path = "data/_histdata"
) -> Path:
    """1 年分の ZIP を落として保存先を返す。既にあれば取り直さない。"""
    try:
        from histdata import download_hist_data
        from histdata.api import Platform, TimeFrame
    except ImportError as exc:  # pragma: no cover - 依存が無い環境向け
        raise HistDataError(
            "histdata パッケージが必要です: pip install histdata"
        ) from exc

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    expected = directory / f"DAT_ASCII_{pair.upper()}_M1_{year}.zip"
    if expected.exists() and expected.stat().st_size > 1000:
        return expected

    try:
        result = download_hist_data(
            year=str(year),
            month=None,
            pair=pair.lower(),
            platform=Platform.GENERIC_ASCII,
            time_frame=TimeFrame.ONE_MINUTE,
            output_directory=str(directory),
            verbose=False,
        )
    except Exception as exc:
        raise HistDataError(f"{pair} {year} の取得に失敗しました: {exc}") from exc
    return Path(result)


def download_month(
    pair: str, year: int, month: int, out_dir: str | Path = "data/_histdata"
) -> Path:
    """当年は年単位で配られないので月単位で取る。"""
    try:
        from histdata import download_hist_data
        from histdata.api import Platform, TimeFrame
    except ImportError as exc:  # pragma: no cover
        raise HistDataError(
            "histdata パッケージが必要です: pip install histdata"
        ) from exc

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    expected = directory / f"DAT_ASCII_{pair.upper()}_M1_{year}{month:02d}.zip"
    if expected.exists() and expected.stat().st_size > 1000:
        return expected

    try:
        result = download_hist_data(
            year=str(year),
            month=str(month),
            pair=pair.lower(),
            platform=Platform.GENERIC_ASCII,
            time_frame=TimeFrame.ONE_MINUTE,
            output_directory=str(directory),
            verbose=False,
        )
    except Exception as exc:
        raise HistDataError(f"{pair} {year}-{month:02d} の取得に失敗しました: {exc}") from exc
    return Path(result)
