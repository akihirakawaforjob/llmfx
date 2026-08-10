"""トレード記録の SQLite ストア.

バックテストとペーパー取引の両方がここへ書き込む。LLM が書いた所感も
同じ行に紐づけて保存し、後から「どの仮説がどれだけ当たったか」を
集計できるようにする。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..domain.types import Trade

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    mode         TEXT NOT NULL,
    instrument   TEXT NOT NULL,
    granularity  TEXT NOT NULL,
    config_json  TEXT NOT NULL,
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             INTEGER REFERENCES runs(id),
    instrument         TEXT NOT NULL,
    side               TEXT NOT NULL,
    units              REAL NOT NULL,
    entry_time         TEXT NOT NULL,
    entry_price        REAL NOT NULL,
    exit_time          TEXT NOT NULL,
    exit_price         REAL NOT NULL,
    stop_loss          REAL NOT NULL,
    take_profit        REAL NOT NULL,
    rr_at_entry        REAL NOT NULL,
    target_source      TEXT NOT NULL,
    pnl                REAL NOT NULL,
    r_multiple         REAL NOT NULL,
    exit_reason        TEXT NOT NULL,
    bars_held          INTEGER NOT NULL,
    equity_after       REAL NOT NULL,
    entry_note_json    TEXT,
    exit_note_json     TEXT,
    gate_json          TEXT,
    structure_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);
CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);

CREATE TABLE IF NOT EXISTS reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    run_id      INTEGER REFERENCES runs(id),
    scope       TEXT NOT NULL,
    stats_json  TEXT NOT NULL,
    report_json TEXT NOT NULL,
    markdown    TEXT NOT NULL
);
"""


@dataclass
class RunHandle:
    run_id: int
    mode: str


class JournalStore:
    def __init__(self, path: str | Path = "data/journal.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "JournalStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    def start_run(
        self, mode: str, instrument: str, granularity: str, config: dict, notes: str | None = None
    ) -> RunHandle:
        cursor = self._conn.execute(
            "INSERT INTO runs (started_at, mode, instrument, granularity, config_json, notes)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                mode,
                instrument,
                granularity,
                json.dumps(config, ensure_ascii=False, default=str),
                notes,
            ),
        )
        self._conn.commit()
        return RunHandle(run_id=int(cursor.lastrowid), mode=mode)

    def record_trade(self, trade: Trade, instrument: str, run_id: int | None = None) -> int:
        structure = None
        if trade.structure is not None:
            structure = {
                "trend": trade.structure.trend.value,
                "last_high": trade.structure.last_high,
                "last_low": trade.structure.last_low,
                "prior_high": trade.structure.prior_high,
                "prior_low": trade.structure.prior_low,
                "last_high_label": trade.structure.last_high_label.value,
                "last_low_label": trade.structure.last_low_label.value,
                "atr": trade.structure.atr,
            }
        cursor = self._conn.execute(
            "INSERT INTO trades ("
            " run_id, instrument, side, units, entry_time, entry_price, exit_time, exit_price,"
            " stop_loss, take_profit, rr_at_entry, target_source, pnl, r_multiple, exit_reason,"
            " bars_held, equity_after, entry_note_json, exit_note_json, gate_json, structure_json"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                instrument,
                trade.side.value,
                trade.units,
                trade.entry_time.isoformat(),
                trade.entry_price,
                trade.exit_time.isoformat(),
                trade.exit_price,
                trade.stop_loss,
                trade.take_profit,
                trade.rr_at_entry,
                trade.target_source,
                trade.pnl,
                trade.r_multiple,
                trade.exit_reason.value,
                trade.bars_held,
                trade.equity_after,
                _dump(trade.entry_note),
                _dump(trade.exit_note),
                _dump(trade.gate_decision),
                _dump(structure),
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def record_trades(
        self, trades: Iterable[Trade], instrument: str, run_id: int | None = None
    ) -> int:
        count = 0
        for trade in trades:
            self.record_trade(trade, instrument, run_id)
            count += 1
        return count

    # ------------------------------------------------------------------
    def recent_trades(
        self, limit: int = 200, since: datetime | None = None, run_id: int | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM trades"
        clauses: list[str] = []
        params: list[Any] = []
        if since is not None:
            clauses.append("exit_time >= ?")
            params.append(since.isoformat())
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY exit_time DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def notes(self, limit: int = 200, since: datetime | None = None) -> list[dict[str, Any]]:
        """所感が記録されているトレードだけを取り出す。"""
        trades = self.recent_trades(limit=limit, since=since)
        collected: list[dict[str, Any]] = []
        for trade in trades:
            if not trade.get("entry_note") and not trade.get("exit_note"):
                continue
            collected.append(
                {
                    "exit_time": trade["exit_time"],
                    "side": trade["side"],
                    "r_multiple": trade["r_multiple"],
                    "exit_reason": trade["exit_reason"],
                    "entry_note": trade.get("entry_note"),
                    "exit_note": trade.get("exit_note"),
                }
            )
        return collected

    def save_review(
        self,
        scope: str,
        stats: dict,
        report: dict,
        markdown: str,
        run_id: int | None = None,
    ) -> int:
        cursor = self._conn.execute(
            "INSERT INTO reviews (created_at, run_id, scope, stats_json, report_json, markdown)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                run_id,
                scope,
                json.dumps(stats, ensure_ascii=False, default=str),
                json.dumps(report, ensure_ascii=False, default=str),
                markdown,
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def trade_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM trades").fetchone()
        return int(row["n"])


def _dump(payload: Any) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, ensure_ascii=False, default=str)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for source, target in (
        ("entry_note_json", "entry_note"),
        ("exit_note_json", "exit_note"),
        ("gate_json", "gate"),
        ("structure_json", "structure"),
    ):
        raw = data.pop(source, None)
        data[target] = json.loads(raw) if raw else None
    return data
