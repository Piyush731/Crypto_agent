"""Timestamped SQLite market store for causal research and paper inference."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=30000;

CREATE TABLE IF NOT EXISTS instruments (
    provider       TEXT NOT NULL,
    instrument_id  TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    asset_type     TEXT NOT NULL,
    settle_currency TEXT,
    contract_value REAL,
    lot_size       REAL,
    tick_size      REAL,
    state          TEXT,
    updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (provider, instrument_id)
);

CREATE TABLE IF NOT EXISTS candles (
    provider       TEXT NOT NULL,
    instrument_id  TEXT NOT NULL,
    timeframe      TEXT NOT NULL,
    ts_ms          INTEGER NOT NULL,
    open           REAL NOT NULL,
    high           REAL NOT NULL,
    low            REAL NOT NULL,
    close          REAL NOT NULL,
    volume         REAL NOT NULL DEFAULT 0,
    quote_volume   REAL NOT NULL DEFAULT 0,
    contracts      REAL NOT NULL DEFAULT 0,
    confirmed      INTEGER NOT NULL CHECK (confirmed IN (0,1)),
    ingested_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (provider, instrument_id, timeframe, ts_ms),
    FOREIGN KEY (provider, instrument_id)
      REFERENCES instruments(provider, instrument_id)
);

CREATE INDEX IF NOT EXISTS idx_candles_lookup
ON candles(provider, instrument_id, timeframe, ts_ms DESC);

CREATE TABLE IF NOT EXISTS funding_rates (
    provider       TEXT NOT NULL,
    instrument_id  TEXT NOT NULL,
    funding_time_ms INTEGER NOT NULL,
    funding_rate   REAL NOT NULL,
    premium        REAL,
    observed_at_ms INTEGER NOT NULL,
    ingested_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (provider, instrument_id, funding_time_ms),
    FOREIGN KEY (provider, instrument_id)
      REFERENCES instruments(provider, instrument_id)
);

CREATE INDEX IF NOT EXISTS idx_funding_lookup
ON funding_rates(provider, instrument_id, funding_time_ms DESC);

CREATE TABLE IF NOT EXISTS open_interest (
    provider       TEXT NOT NULL,
    instrument_id  TEXT NOT NULL,
    ts_ms          INTEGER NOT NULL,
    oi_contracts   REAL NOT NULL,
    oi_base        REAL,
    oi_usd         REAL,
    ingested_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (provider, instrument_id, ts_ms),
    FOREIGN KEY (provider, instrument_id)
      REFERENCES instruments(provider, instrument_id)
);

CREATE INDEX IF NOT EXISTS idx_oi_lookup
ON open_interest(provider, instrument_id, ts_ms DESC);
"""


class MarketStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def upsert_instrument(self, provider: str, row: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO instruments (
                    provider, instrument_id, symbol, asset_type,
                    settle_currency, contract_value, lot_size,
                    tick_size, state, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(provider, instrument_id) DO UPDATE SET
                    symbol=excluded.symbol,
                    asset_type=excluded.asset_type,
                    settle_currency=excluded.settle_currency,
                    contract_value=excluded.contract_value,
                    lot_size=excluded.lot_size,
                    tick_size=excluded.tick_size,
                    state=excluded.state,
                    updated_at=datetime('now')
                """,
                (
                    provider,
                    row["instrument_id"],
                    row["symbol"],
                    row.get("asset_type", "swap"),
                    row.get("settle_currency"),
                    row.get("contract_value"),
                    row.get("lot_size"),
                    row.get("tick_size"),
                    row.get("state"),
                ),
            )

    def upsert_candles(
        self,
        provider: str,
        instrument_id: str,
        timeframe: str,
        frame: pd.DataFrame,
    ) -> int:
        if frame.empty:
            return 0
        rows = []
        for timestamp, row in frame.iterrows():
            ts = pd.Timestamp(timestamp)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            rows.append(
                (
                    provider,
                    instrument_id,
                    timeframe,
                    int(ts.timestamp() * 1000),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row.get("volume", 0)),
                    float(row.get("quote_volume", 0)),
                    float(row.get("contracts", 0)),
                    int(bool(row.get("confirmed", True))),
                )
            )

        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO candles (
                    provider, instrument_id, timeframe, ts_ms,
                    open, high, low, close, volume,
                    quote_volume, contracts, confirmed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, instrument_id, timeframe, ts_ms)
                DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume,
                    quote_volume=excluded.quote_volume,
                    contracts=excluded.contracts,
                    confirmed=excluded.confirmed,
                    ingested_at=datetime('now')
                """,
                rows,
            )
        return len(rows)

    def upsert_funding(self, provider: str, instrument_id: str, row: dict) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO funding_rates (
                    provider, instrument_id, funding_time_ms,
                    funding_rate, premium, observed_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, instrument_id, funding_time_ms)
                DO UPDATE SET
                    funding_rate=excluded.funding_rate,
                    premium=excluded.premium,
                    observed_at_ms=excluded.observed_at_ms,
                    ingested_at=datetime('now')
                """,
                (
                    provider,
                    instrument_id,
                    int(row["funding_time_ms"]),
                    float(row["funding_rate"]),
                    row.get("premium"),
                    int(row["observed_at_ms"]),
                ),
            )

    def upsert_open_interest(
        self, provider: str, instrument_id: str, row: dict
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO open_interest (
                    provider, instrument_id, ts_ms,
                    oi_contracts, oi_base, oi_usd
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, instrument_id, ts_ms)
                DO UPDATE SET
                    oi_contracts=excluded.oi_contracts,
                    oi_base=excluded.oi_base,
                    oi_usd=excluded.oi_usd,
                    ingested_at=datetime('now')
                """,
                (
                    provider,
                    instrument_id,
                    int(row["ts_ms"]),
                    float(row["oi_contracts"]),
                    row.get("oi_base"),
                    row.get("oi_usd"),
                ),
            )

    def get_candles(
        self,
        provider: str,
        instrument_id: str,
        timeframe: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        completed_only: bool = True,
    ) -> pd.DataFrame:
        conditions = [
            "provider=?", "instrument_id=?", "timeframe=?"
        ]
        params: list = [provider, instrument_id, timeframe]
        if start_ms is not None:
            conditions.append("ts_ms>=?")
            params.append(int(start_ms))
        if end_ms is not None:
            conditions.append("ts_ms<=?")
            params.append(int(end_ms))
        if completed_only:
            conditions.append("confirmed=1")

        query = f"""
            SELECT ts_ms, open, high, low, close,
                   volume, quote_volume, contracts, confirmed
            FROM candles
            WHERE {' AND '.join(conditions)}
            ORDER BY ts_ms ASC
        """
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "open", "high", "low", "close", "volume",
                    "quote_volume", "contracts", "confirmed",
                ]
            )
        frame = pd.DataFrame([dict(row) for row in rows])
        frame["timestamp"] = pd.to_datetime(frame.pop("ts_ms"), unit="ms", utc=True)
        return frame.set_index("timestamp")

    def table_counts(self) -> dict[str, int]:
        with self.connect() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in [
                    "instruments", "candles", "funding_rates", "open_interest"
                ]
            }
