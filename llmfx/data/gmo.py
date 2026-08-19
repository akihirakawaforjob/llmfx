"""GMOコイン Public API からローソク足を取得する.

口座も API キーも要らずに過去のローソク足が取れるため、口座開設の判断より前に
実データでバックテストを回せる。これが OANDA との一番の違い。

エンドポイント:
    GET https://api.coin.z.com/public/v1/klines?symbol=&interval=&date=

日付の区切りが特殊:
    intraday(1min〜1hour)の `date` は YYYYMMDD だが、区切りは日本時間の朝 6:00。
    つまり date=20260817 が返すのは UTC で 2026-08-16 21:00 〜 2026-08-17 20:45。
    そのため要求期間の前後に 1 日ずつ余裕を持って取得し、openTime で重複を
    落としてから範囲で切る。境界の取りこぼしを防ぐのはこの実装の要点。

銘柄:
    BTC_JPY 等 = 暗号資産FX(レバレッジ)。取引手数料 0、コストはスプレッドと
                 レバレッジ手数料 0.04%/日
    BTC 等     = 取引所現物。Taker 0.05% / Maker -0.01%
どちらも klines は同じ形式で取れる。
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterator

import httpx

from ..domain.types import Candle

PUBLIC_HOST = "https://api.coin.z.com/public"

# date に YYYYMMDD を渡す足。これ以外(4hour 以上)は YYYY を渡す。
INTRADAY_INTERVALS = ("1min", "5min", "10min", "15min", "30min", "1hour")
LONG_INTERVALS = ("4hour", "8hour", "12hour", "1day", "1week", "1month")

# 公式ドキュメントの記載。これより前を要求しても Not found が返る。
EARLIEST_INTRADAY = date(2021, 4, 15)

# OANDA 形式(M15 等)から GMO 形式への対応表。設定 YAML をそのまま使えるように。
_GRANULARITY_ALIASES = {
    "M1": "1min",
    "M5": "5min",
    "M10": "10min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1hour",
    "H4": "4hour",
    "H8": "8hour",
    "H12": "12hour",
    "D": "1day",
    "D1": "1day",
    "W": "1week",
    "W1": "1week",
}


class GmoError(RuntimeError):
    pass


def to_gmo_interval(granularity: str) -> str:
    """M15 のような表記も 15min のような GMO 表記も受け付ける。"""
    text = granularity.strip()
    if text in INTRADAY_INTERVALS or text in LONG_INTERVALS:
        return text
    interval = _GRANULARITY_ALIASES.get(text.upper())
    if interval is None:
        raise GmoError(
            f"未対応の足です: {granularity!r}。"
            f"指定できるのは {', '.join(INTRADAY_INTERVALS + LONG_INTERVALS)} "
            f"または M1/M5/M15/M30/H1/H4/D です。"
        )
    return interval


def _date_keys(interval: str, start: datetime, end: datetime) -> list[str]:
    """要求期間を覆うのに必要な `date` パラメータを列挙する。

    intraday は区切りが JST 6:00 なので、前後 1 日ずつ広げて取りこぼしを防ぐ。
    重複分は openTime で落とすため、多めに取って構わない。
    """
    if interval in LONG_INTERVALS:
        return [str(y) for y in range(start.year, end.year + 1)]

    first = max((start - timedelta(days=1)).date(), EARLIEST_INTRADAY)
    last = (end + timedelta(days=1)).date()
    if first > last:
        return []
    keys: list[str] = []
    cursor = first
    while cursor <= last:
        keys.append(cursor.strftime("%Y%m%d"))
        cursor += timedelta(days=1)
    return keys


def _parse_rows(rows: list[dict]) -> Iterator[tuple[int, Candle]]:
    for row in rows:
        open_time = int(row["openTime"])
        yield open_time, Candle(
            time=datetime.fromtimestamp(open_time / 1000.0, tz=timezone.utc),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume") or 0.0),
        )


def _request(client: httpx.Client, params: dict, retries: int) -> list[dict]:
    """1 リクエスト分。status != 0 は API 側のエラーなので中身を見て判断する。"""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.get("/v1/klines", params=params)
            # 未来日付やデータの無い日は 404 が返る(JSON のエラーではない)。
            # 期間の末尾は JST 6:00 区切りの都合で 1 日先まで問い合わせるため、
            # ここを例外にすると「今日まで」の取得が必ず最後で落ちる。
            if response.status_code == 404:
                return []
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2.0**attempt)
                continue
            raise GmoError(f"取得に失敗しました({params}): {exc}") from exc

        if payload.get("status") != 0:
            messages = payload.get("messages") or []
            codes = {m.get("message_code") for m in messages}
            # データが無い日(上場前・メンテ日)はエラーではなく空として扱う。
            if "ERR-5207" in codes:
                return []
            raise GmoError(f"GMO API がエラーを返しました({params}): {messages}")
        return payload.get("data") or []

    raise GmoError(f"取得に失敗しました({params}): {last_error}")


def fetch_klines(
    symbol: str,
    granularity: str,
    start: datetime,
    end: datetime,
    *,
    client: httpx.Client | None = None,
    retries: int = 3,
    pause: float = 0.12,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> list[Candle]:
    """`start` 以上 `end` 未満の確定足を返す(UTC、openTime の昇順)。

    `on_progress(完了リクエスト数, 全リクエスト数, 取得済み本数)` を渡すと進捗を
    受け取れる。数年分だと 1,000 リクエストを超えるため、進捗表示は実質必須。
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise GmoError("start / end はタイムゾーン付きで渡してください")
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    if start >= end:
        raise GmoError(f"期間が逆転しています: {start} >= {end}")

    interval = to_gmo_interval(granularity)
    keys = _date_keys(interval, start, end)
    if not keys:
        raise GmoError(
            f"指定期間に取得できる日がありません。intraday の取得可能開始日は "
            f"{EARLIEST_INTRADAY:%Y-%m-%d} です。"
        )

    owned = client is None
    session = client or httpx.Client(base_url=PUBLIC_HOST, timeout=30.0)
    # openTime をキーにすることで、日付境界での重複をそのまま吸収できる。
    collected: dict[int, Candle] = {}
    try:
        for index, key in enumerate(keys, start=1):
            rows = _request(
                session, {"symbol": symbol, "interval": interval, "date": key}, retries
            )
            for open_time, candle in _parse_rows(rows):
                collected[open_time] = candle
            if on_progress is not None:
                on_progress(index, len(keys), len(collected))
            if pause and index < len(keys):
                time.sleep(pause)
    finally:
        if owned:
            session.close()

    candles = [c for _, c in sorted(collected.items())]
    return [c for c in candles if start <= c.time < end]


def fetch_symbols(client: httpx.Client | None = None) -> list[dict]:
    """取扱銘柄と最小注文数量・呼値・手数料。pip_size を決めるのに使う。"""
    owned = client is None
    session = client or httpx.Client(base_url=PUBLIC_HOST, timeout=30.0)
    try:
        response = session.get("/v1/symbols")
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise GmoError(f"銘柄一覧の取得に失敗しました: {exc}") from exc
    finally:
        if owned:
            session.close()

    if payload.get("status") != 0:
        raise GmoError(f"GMO API がエラーを返しました: {payload.get('messages')}")
    return payload.get("data") or []
