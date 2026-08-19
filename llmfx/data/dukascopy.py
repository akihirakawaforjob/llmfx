"""Dukascopy の公開データフィードから FX の 1 分足を取得する.

口座も API キーも要らず、2003 年頃から現在までの 1 分足が取れる。
GMOコインが暗号資産専用で、国内の FX 業者に一般向けの公開 API が無いため、
FX の検証データはここから引く。バックテストのデータ源は発注先と別で構わない。

URL の形:
    https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YYYY}/{MM}/{DD}/{PRICE}_candles_min_1.bi5

**月は 0 始まり**。1 月が 00、12 月が 11。ここを間違えると 1 か月ずれた
データを黙って掴むことになる。

ファイルの中身:
    生 LZMA(FORMAT_ALONE)で圧縮された 24 バイト固定長レコードの並び。
    ビッグエンディアンで `>iiiiif`:

        int32   その日の 00:00 UTC からの経過秒
        int32   始値, 終値, 安値, 高値   ← 高安が後ろにある点に注意
        float32 出来高

    価格は整数。USDJPY のような JPY クオートは 1000 で、それ以外は
    100000 で割ると実際の価格になる。

その他の癖:
    - 土日・祝日はファイルが無く 404 が返る。エラーではなく「データ無し」
    - 503 が断続的に混ざる。リトライ前提で組む必要がある
    - 取引の無かった分は価格 0 のレコードとして詰まっている。読み飛ばす
"""

from __future__ import annotations

import lzma
import struct
import time
from datetime import datetime, timedelta, timezone
from typing import Callable

import httpx

from ..domain.types import Candle

DATAFEED_HOST = "https://datafeed.dukascopy.com/datafeed"
RECORD = struct.Struct(">iiiiif")
RECORD_SIZE = RECORD.size  # 24

# 公開されている最古はおおむね 2003 年。これより前は 404 になる。
EARLIEST = datetime(2003, 5, 5, tzinfo=timezone.utc)


class DukascopyError(RuntimeError):
    pass


def point_scale(symbol: str) -> int:
    """整数価格を実価格へ直すための除数。

    JPY クオートは小数 3 桁(0.001 刻み)、それ以外は 5 桁(0.00001 刻み)。
    """
    return 1000 if symbol.upper().endswith("JPY") else 100_000


def _url(symbol: str, day: datetime, price: str) -> str:
    # 月は 0 始まり。1 月 = 00。
    return (
        f"{DATAFEED_HOST}/{symbol.upper()}/{day.year:04d}/{day.month - 1:02d}/"
        f"{day.day:02d}/{price.upper()}_candles_min_1.bi5"
    )


def decode_bi5(payload: bytes, day: datetime, scale: int) -> list[Candle]:
    """1 日分の bi5 を展開してローソク足へ直す。"""
    if not payload:
        return []
    try:
        data = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(payload)
    except lzma.LZMAError as exc:
        raise DukascopyError(f"bi5 を展開できません({day:%Y-%m-%d}): {exc}") from exc

    if len(data) % RECORD_SIZE:
        raise DukascopyError(
            f"レコード長が合いません({day:%Y-%m-%d}): {len(data)} バイトは "
            f"{RECORD_SIZE} の倍数ではありません"
        )

    candles: list[Candle] = []
    for offset in range(0, len(data), RECORD_SIZE):
        seconds, open_, close, low, high, volume = RECORD.unpack_from(data, offset)
        # 取引の無かった分は価格 0 で詰まっている。足として扱わない。
        if open_ == 0 and close == 0 and low == 0 and high == 0:
            continue
        candles.append(
            Candle(
                time=day + timedelta(seconds=seconds),
                open=open_ / scale,
                high=high / scale,
                low=low / scale,
                close=close / scale,
                volume=float(volume),
            )
        )
    return candles


MAX_BACKOFF = 30.0
"""指数バックオフの上限(秒)。青天井にすると 1 日分で何分も待つことになる。"""


class _Unavailable(Exception):
    """リトライを尽くしても取れなかった日。呼び出し側が握るか落とすかを決める。"""


def _download(
    client: httpx.Client, url: str, retries: int, backoff: float
) -> bytes | None:
    """1 日分を取る。データが無い日は None。取れなかった日は _Unavailable。"""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(min(backoff * (2 ** (attempt - 1)), MAX_BACKOFF))
        try:
            response = client.get(url)
        except httpx.HTTPError as exc:
            last_error = exc
            continue

        # 土日・祝日・上場前はファイルが存在しない。
        if response.status_code == 404:
            return None
        # 503 は断続的にも連続的にも来る。同じ日に対して何度も返ることがあり、
        # ここで諦めると数年分の取得が 1 日のせいで丸ごと落ちる。
        if response.status_code in (429, 500, 502, 503, 504):
            last_error = DukascopyError(f"HTTP {response.status_code}")
            continue
        if response.status_code != 200:
            raise DukascopyError(f"取得に失敗しました({url}): HTTP {response.status_code}")
        return response.content

    raise _Unavailable(f"{url}: {last_error}")


def fetch_m1_candles(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    price: str = "BID",
    client: httpx.Client | None = None,
    retries: int = 8,
    backoff: float = 1.0,
    pause: float = 0.05,
    on_progress: Callable[[int, int, int], None] | None = None,
    strict: bool = True,
    failures: list[datetime] | None = None,
) -> list[Candle]:
    """`start` 以上 `end` 未満の 1 分足を返す(UTC 昇順)。

    `on_progress(完了日数, 全日数, 取得済み本数)` で進捗を受け取れる。
    数年分だと 1,000 日を超えるため、進捗表示は実質必須。

    `strict=True`(既定)なら、リトライを尽くしても取れない日があった時点で
    落とす。`strict=False` ならその日を飛ばして続け、日付を `failures` へ
    積む。**飛ばした日を黙って無かったことにはしない。**穴の空いたデータで
    バックテストを回すと、理由の分からない成績差として現れるため。
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise DukascopyError("start / end はタイムゾーン付きで渡してください")
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    if start >= end:
        raise DukascopyError(f"期間が逆転しています: {start} >= {end}")
    if end <= EARLIEST:
        raise DukascopyError(f"取得可能なのは {EARLIEST:%Y-%m-%d} 以降です")

    scale = point_scale(symbol)
    first = max(start, EARLIEST).replace(hour=0, minute=0, second=0, microsecond=0)
    days: list[datetime] = []
    cursor = first
    while cursor < end:
        days.append(cursor)
        cursor += timedelta(days=1)

    owned = client is None
    session = client or httpx.Client(
        timeout=60.0, headers={"User-Agent": "llmfx/1.0"}, follow_redirects=True
    )
    collected: list[Candle] = []
    try:
        for index, day in enumerate(days, start=1):
            try:
                payload = _download(session, _url(symbol, day, price), retries, backoff)
            except _Unavailable as exc:
                if strict:
                    raise DukascopyError(f"取得に失敗しました({exc})") from exc
                if failures is not None:
                    failures.append(day)
                payload = None
            if payload is not None:
                collected.extend(decode_bi5(payload, day, scale))
            if on_progress is not None:
                on_progress(index, len(days), len(collected))
            if pause and index < len(days):
                time.sleep(pause)
    finally:
        if owned:
            session.close()

    collected.sort(key=lambda c: c.time)
    return [c for c in collected if start <= c.time < end]
