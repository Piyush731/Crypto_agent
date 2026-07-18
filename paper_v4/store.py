"""Persistent, restart-safe SQLite ledger for the v4 forward paper runtime."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=30000;

CREATE TABLE IF NOT EXISTS runtime_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    direction INTEGER NOT NULL CHECK(direction IN (-1, 1)),
    signal_time TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity REAL NOT NULL,
    notional REAL NOT NULL,
    stop_loss REAL NOT NULL,
    entry_fee REAL NOT NULL,
    funding_cost REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    direction INTEGER NOT NULL,
    signal_time TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity REAL NOT NULL,
    notional REAL NOT NULL,
    stop_loss REAL NOT NULL,
    entry_fee REAL NOT NULL,
    funding_cost REAL NOT NULL,
    exit_time TEXT NOT NULL,
    exit_price REAL NOT NULL,
    exit_fee REAL NOT NULL,
    gross_pnl REAL NOT NULL,
    net_pnl REAL NOT NULL,
    exit_reason TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    bar_time TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    capital REAL NOT NULL,
    position_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS processed_signals (
    signal_time TEXT PRIMARY KEY,
    execution_time TEXT NOT NULL,
    long_symbol TEXT,
    short_symbol TEXT
);

CREATE TABLE IF NOT EXISTS notification_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_key TEXT NOT NULL UNIQUE,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    sent_at TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def get_state(self, key: str, default: Any = None) -> Any:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value FROM runtime_state WHERE key=?", (key,)
            ).fetchone()
        return default if row is None else json.loads(row["value"])

    def set_state(self, connection: sqlite3.Connection, key: str, value: Any) -> None:
        connection.execute(
            """
            INSERT INTO runtime_state(key, value, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, json.dumps(value), utc_now()),
        )

    def positions(self, connection: sqlite3.Connection | None = None) -> dict[str, dict]:
        owns = connection is None
        connection = connection or self.connect()
        try:
            rows = connection.execute("SELECT * FROM positions ORDER BY symbol").fetchall()
            return {row["symbol"]: dict(row) for row in rows}
        finally:
            if owns:
                connection.close()

    def queue(self, connection: sqlite3.Connection, event_key: str, message: str) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO notification_outbox(event_key, message, created_at)
            VALUES (?, ?, ?)
            """,
            (event_key, message, utc_now()),
        )

    def pending_notifications(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM notification_outbox
                   WHERE sent_at IS NULL ORDER BY id LIMIT 100"""
            ).fetchall()
        return [dict(row) for row in rows]

    def notification_sent(self, notification_id: int) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE notification_outbox SET sent_at=?, attempts=attempts+1 WHERE id=?",
                (utc_now(), notification_id),
            )

    def notification_failed(self, notification_id: int, error: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE notification_outbox
                   SET attempts=attempts+1, last_error=? WHERE id=?""",
                (error[:500], notification_id),
            )

    def summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            capital = self.get_state("capital", 10_000.0)
            trades = connection.execute(
                "SELECT COUNT(*) count, COALESCE(SUM(net_pnl),0) pnl FROM trades"
            ).fetchone()
            positions = connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0]
            last_equity = connection.execute(
                "SELECT equity FROM equity_snapshots ORDER BY bar_time DESC LIMIT 1"
            ).fetchone()
        return {
            "capital": float(capital),
            "equity": float(last_equity[0]) if last_equity else float(capital),
            "trades": int(trades["count"]),
            "realized_pnl": float(trades["pnl"]),
            "positions": int(positions),
            "last_processed_bar": self.get_state("last_processed_bar"),
        }
