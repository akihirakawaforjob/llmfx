"""Claude API クライアントのラッパー.

設計上の要点:

  - API キーが無ければ `available == False` になり、呼び出し側は
    ルールベースのみで動作を継続する(例外で落とさない)
  - システムプロンプトは prompt caching の対象にする。売買ルールの
    説明文は毎回同じなので、キャッシュが効けば実質無料になる
  - バックテストで LLM を使う場合に備え、応答を SQLite にキャッシュする。
    同じ入力なら同じ出力になり、再実行の結果が変わらない
  - `stop_reason == "refusal"` を必ず確認してから content を読む
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import LLMConfig
from .schemas import json_schema_for

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """LLM が使えない状態(キー未設定・SDK 未導入・API 障害)。"""


class ResponseCache:
    """入力ハッシュ → JSON 応答の永続キャッシュ。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "  key TEXT PRIMARY KEY,"
            "  payload TEXT NOT NULL,"
            "  created_at TEXT DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        self._conn.commit()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, payload: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, payload) VALUES (?, ?)",
                (key, json.dumps(payload, ensure_ascii=False)),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class LLMClient:
    def __init__(self, config: LLMConfig, api_key: str | None = None) -> None:
        self.config = config
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client: Any = None
        self._cache: ResponseCache | None = None
        self._unavailable_reason: str | None = None
        self._use_refusal_fallback = True

        if not config.enabled:
            self._unavailable_reason = "設定で LLM が無効化されています (llm.enabled: false)"
        elif not self._api_key:
            self._unavailable_reason = "ANTHROPIC_API_KEY が未設定です"

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        if self._unavailable_reason is not None:
            return False
        return self._ensure_client() is not None

    @property
    def unavailable_reason(self) -> str | None:
        if self._unavailable_reason is None:
            return None
        return self._unavailable_reason

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError:  # pragma: no cover - 依存が入っていれば通らない
            self._unavailable_reason = "anthropic パッケージが導入されていません"
            return None
        try:
            self._client = anthropic.Anthropic(
                api_key=self._api_key, timeout=self.config.timeout_seconds
            )
        except Exception as exc:  # pragma: no cover
            self._unavailable_reason = f"Anthropic クライアントの初期化に失敗: {exc}"
            return None
        return self._client

    def _ensure_cache(self) -> ResponseCache:
        if self._cache is None:
            self._cache = ResponseCache(self.config.cache_path)
        return self._cache

    # ------------------------------------------------------------------
    def structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        use_cache: bool = True,
        max_tokens: int | None = None,
    ) -> T:
        """構造化出力を 1 回取得する。失敗時は LLMUnavailable を送出する。"""
        if not self.available:
            raise LLMUnavailable(self.unavailable_reason or "LLM が利用できません")

        cache_key = _cache_key(
            model=self.config.model,
            effort=self.config.effort,
            system=system,
            user=user,
            schema_name=schema.__name__,
        )
        if use_cache:
            cached = self._ensure_cache().get(cache_key)
            if cached is not None:
                try:
                    return schema.model_validate(cached)
                except ValidationError:
                    logger.debug("キャッシュ内容がスキーマ不一致のため破棄します")

        payload = self._request(system, user, schema, max_tokens)
        if use_cache:
            self._ensure_cache().put(cache_key, payload)
        return schema.model_validate(payload)

    def _request(
        self,
        system: str,
        user: str,
        schema: type[BaseModel],
        max_tokens: int | None,
    ) -> dict:
        import anthropic

        client = self._ensure_client()
        params: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens or self.config.max_tokens,
            # 売買ルールの説明は毎回同一なのでキャッシュ対象にする。
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "effort": self.config.effort,
                "format": {
                    "type": "json_schema",
                    "schema": json_schema_for(schema),
                },
            },
        }

        try:
            response = self._create(client, params)
        except anthropic.APIStatusError as exc:
            raise LLMUnavailable(f"Claude API エラー ({exc.status_code}): {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailable(f"Claude API へ接続できません: {exc}") from exc

        # 安全性の判断で拒否された場合、content は空か部分的なので先に確認する。
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMUnavailable(f"Claude がリクエストを拒否しました (category={category})")
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise LLMUnavailable(
                "応答が max_tokens で打ち切られました。llm.max_tokens を増やしてください"
            )

        text = next(
            (block.text for block in response.content if getattr(block, "type", None) == "text"),
            None,
        )
        if not text:
            raise LLMUnavailable("Claude の応答にテキストブロックがありません")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"構造化出力の JSON 解析に失敗: {exc}") from exc

    def _create(self, client: Any, params: dict[str, Any]) -> Any:
        """安全性拒否のフォールバックつきでリクエストする。

        `fallbacks` はベータ機能なので、利用できないアカウントでは
        400 が返る。その場合は通常エンドポイントへ静かに切り替える。
        """
        import anthropic

        if self._use_refusal_fallback:
            try:
                return client.beta.messages.create(
                    **params,
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            except anthropic.BadRequestError as exc:
                logger.info("server-side fallback が利用できないため通常経路に切り替えます: %s", exc)
                self._use_refusal_fallback = False
            except (AttributeError, TypeError) as exc:
                logger.info("SDK が fallbacks 引数に未対応のため通常経路に切り替えます: %s", exc)
                self._use_refusal_fallback = False
        return client.messages.create(**params)

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()
            self._cache = None


def _cache_key(**parts: str) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
