"""OANDA v20 REST クライアント(ローソク足取得 + 発注).

安全側の設計:
  - 既定はデモ環境(api-fxpractice)。本番環境を使うには環境変数で
    OANDA_ENV=live を明示し、さらに allow_live=True を渡す必要がある。
  - 未確定足(complete=false)は返さない。バックテストと同じく確定足のみを扱う。

注意: この環境では OANDA の資格情報が無いため、接続テストは未実施。
初回利用時はデモ口座で `llmfx data fetch` から動作確認すること。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from ..domain.types import Candle

PRACTICE_HOST = "https://api-fxpractice.oanda.com"
LIVE_HOST = "https://api-fxtrade.oanda.com"
MAX_CANDLES_PER_REQUEST = 5000


class OandaError(RuntimeError):
    pass


class OandaClient:
    def __init__(
        self,
        api_token: str | None = None,
        account_id: str | None = None,
        environment: str | None = None,
        allow_live: bool = False,
        timeout: float = 30.0,
    ) -> None:
        self.api_token = api_token or os.getenv("OANDA_API_TOKEN")
        self.account_id = account_id or os.getenv("OANDA_ACCOUNT_ID")
        self.environment = (environment or os.getenv("OANDA_ENV") or "practice").lower()

        if not self.api_token:
            raise OandaError(
                "OANDA_API_TOKEN が設定されていません(.env.example を参照)"
            )
        if self.environment == "live" and not allow_live:
            raise OandaError(
                "本番環境(live)での実行は allow_live=True の明示が必要です。"
                "まずデモ口座で十分に検証してください。"
            )
        self.host = LIVE_HOST if self.environment == "live" else PRACTICE_HOST
        self._client = httpx.Client(
            base_url=self.host,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                "Accept-Datetime-Format": "RFC3339",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OandaClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise OandaError(f"OANDA への接続に失敗しました: {exc}") from exc
        if response.status_code >= 400:
            raise OandaError(
                f"OANDA API エラー {response.status_code}: {response.text[:500]}"
            )
        return response.json()

    # ------------------------------------------------------------------
    def fetch_candles(
        self,
        instrument: str,
        granularity: str = "M15",
        count: int = 500,
        price: str = "M",
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> list[Candle]:
        """確定足のみを返す。count が 5000 を超える場合は自動でページングする。"""
        remaining = count
        collected: list[Candle] = []
        cursor_to = to_time

        while remaining > 0:
            batch = min(remaining, MAX_CANDLES_PER_REQUEST)
            params: dict[str, Any] = {
                "granularity": granularity,
                "price": price,
                "count": batch,
            }
            if from_time is not None and cursor_to is None:
                params["from"] = _rfc3339(from_time)
                params.pop("count", None)
                params["count"] = batch
            if cursor_to is not None:
                params["to"] = _rfc3339(cursor_to)

            payload = self._request(
                "GET", f"/v3/instruments/{instrument}/candles", params=params
            )
            candles = _parse_candles(payload, price)
            if not candles:
                break

            collected = candles + collected
            remaining -= len(candles)
            cursor_to = candles[0].time
            if len(candles) < batch:
                break

        collected.sort(key=lambda c: c.time)
        # 重複除去(ページ境界で同じ足が返ることがある)
        deduped: list[Candle] = []
        seen: set[datetime] = set()
        for candle in collected:
            if candle.time in seen:
                continue
            seen.add(candle.time)
            deduped.append(candle)
        return deduped[-count:]

    def latest_candles(
        self, instrument: str, granularity: str, count: int = 200, price: str = "M"
    ) -> list[Candle]:
        return self.fetch_candles(instrument, granularity, count=count, price=price)

    # ------------------------------------------------------------------
    def account_summary(self) -> dict:
        self._require_account()
        return self._request("GET", f"/v3/accounts/{self.account_id}/summary")

    def account_equity(self) -> float:
        summary = self.account_summary()
        return float(summary["account"]["NAV"])

    def open_positions(self) -> list[dict]:
        self._require_account()
        payload = self._request("GET", f"/v3/accounts/{self.account_id}/openPositions")
        return payload.get("positions", [])

    def place_market_order(
        self,
        instrument: str,
        units: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        price_precision: int = 5,
        client_tag: str | None = None,
    ) -> dict:
        """成行注文 + SL/TP を同時に送る(約定と同時に保護が入る)。"""
        self._require_account()
        order: dict[str, Any] = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(int(units)),
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
        if stop_loss is not None:
            order["stopLossOnFill"] = {
                "price": f"{stop_loss:.{price_precision}f}",
                "timeInForce": "GTC",
            }
        if take_profit is not None:
            order["takeProfitOnFill"] = {
                "price": f"{take_profit:.{price_precision}f}",
                "timeInForce": "GTC",
            }
        if client_tag:
            order["clientExtensions"] = {"tag": client_tag, "comment": "llmfx"}

        return self._request(
            "POST", f"/v3/accounts/{self.account_id}/orders", json={"order": order}
        )

    def close_position(self, instrument: str, side: str) -> dict:
        self._require_account()
        body = {"longUnits": "ALL"} if side == "long" else {"shortUnits": "ALL"}
        return self._request(
            "PUT",
            f"/v3/accounts/{self.account_id}/positions/{instrument}/close",
            json=body,
        )

    def _require_account(self) -> None:
        if not self.account_id:
            raise OandaError("OANDA_ACCOUNT_ID が設定されていません")


def _rfc3339(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def parse_oanda_time(value: str) -> datetime:
    """OANDA の RFC3339(小数点以下 9 桁)を datetime へ.

    `datetime.fromisoformat` は小数部 3 桁か 6 桁しか受け付けないため、
    ナノ秒表記をマイクロ秒へ丸めてから渡す。
    """
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1]
    fraction = ""
    if "." in text:
        text, fraction = text.split(".", 1)
        # タイムゾーンオフセットが小数部の後ろに付くケースに備える
        for sign in ("+", "-"):
            if sign in fraction:
                fraction, _offset = fraction.split(sign, 1)
                break
        fraction = fraction[:6].ljust(6, "0")
        text = f"{text}.{fraction}"
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def _parse_candles(payload: dict, price: str) -> list[Candle]:
    key = {"M": "mid", "B": "bid", "A": "ask"}.get(price.upper(), "mid")
    candles: list[Candle] = []
    for raw in payload.get("candles", []):
        if not raw.get("complete", False):
            continue  # 未確定足は使わない
        ohlc = raw[key]
        candles.append(
            Candle(
                time=parse_oanda_time(raw["time"]),
                open=float(ohlc["o"]),
                high=float(ohlc["h"]),
                low=float(ohlc["l"]),
                close=float(ohlc["c"]),
                volume=float(raw.get("volume", 0)),
            )
        )
    return candles
