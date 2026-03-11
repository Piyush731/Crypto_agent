"""
core/db.py
SQLite database manager for the crypto futures agent.

Usage in any module:
    from core.db import get_db
    db = get_db()
    db.save_signal({...})
    db.save_trade({...})
    trades = db.get_trades(symbol="BTCUSDT")

Tables:
  - signals:     Every prediction generated (direction, confidence, components)
  - trades:      Paper/live trade records (entry, exit, PnL)
  - performance: Daily performance snapshots
  - errors:      Error tracking for debugging

Features:
  - Thread-safe with locks
  - WAL mode for concurrent reads
  - Singleton via get_db()
  - Dict-like row access (row["column_name"])
  - Risk queries for circuit breakers
  - Auto-migration for schema upgrades

v3.0.2:
  - Added combined_score to signals table
  - Fixed save_signal log line (confidence display)
  - Migration adds combined_score to existing DBs
"""

import sqlite3
import threading
import traceback as tb_module
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any

# ── Import config ────────────────────────────────────────────
try:
    from config import BASE_DIR
    DB_PATH = BASE_DIR / "agent.db"
except ImportError:
    DB_PATH = Path(__file__).parent.parent / "agent.db"

from core.logger import get_logger

logger = get_logger(__name__)


# ╔═══════════════════════════════════════════════════════════╗
# ║                     TABLE SCHEMAS                        ║
# ╚═══════════════════════════════════════════════════════════╝

SCHEMA_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
    symbol          TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    confidence      REAL    NOT NULL DEFAULT 0,
    combined_score  REAL    DEFAULT 0,
    entry_price     REAL,
    stop_loss       REAL,
    take_profit     REAL,
    timeframe       TEXT    DEFAULT '4h',
    horizon_hours   INTEGER DEFAULT 24,

    ml_direction    TEXT,
    ml_confidence   REAL    DEFAULT 0,
    ml_agreement    REAL    DEFAULT 0,
    sentiment_score REAL    DEFAULT 0,
    ai_direction    TEXT,
    ai_confidence   REAL    DEFAULT 0,
    funding_signal  REAL    DEFAULT 0,
    market_signal   REAL    DEFAULT 0,

    status          TEXT    DEFAULT 'active',
    reject_reason   TEXT,
    notes           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

SCHEMA_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id       INTEGER,
    symbol          TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    mode            TEXT    NOT NULL DEFAULT 'backtest',

    entry_price     REAL    NOT NULL,
    entry_time      TEXT    NOT NULL,
    exit_price      REAL,
    exit_time       TEXT,
    quantity        REAL    DEFAULT 0,
    leverage        REAL    DEFAULT 1,

    stop_loss       REAL,
    take_profit     REAL,

    position_size_usd REAL  DEFAULT 0,
    signal_type     TEXT,
    confidence      REAL    DEFAULT 0,
    margin_used     REAL    DEFAULT 0,

    pnl_usd        REAL    DEFAULT 0,
    pnl_percent     REAL    DEFAULT 0,
    commission      REAL    DEFAULT 0,

    status          TEXT    DEFAULT 'open',
    exit_reason     TEXT,

    notes           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),

    FOREIGN KEY (signal_id) REFERENCES signals(id)
);
"""

SCHEMA_PERFORMANCE = """
CREATE TABLE IF NOT EXISTS performance (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL,
    mode            TEXT    NOT NULL DEFAULT 'backtest',
    symbol          TEXT    DEFAULT 'ALL',

    total_trades    INTEGER DEFAULT 0,
    winning_trades  INTEGER DEFAULT 0,
    losing_trades   INTEGER DEFAULT 0,
    win_rate        REAL    DEFAULT 0,

    total_pnl       REAL    DEFAULT 0,
    avg_pnl         REAL    DEFAULT 0,
    best_trade      REAL    DEFAULT 0,
    worst_trade     REAL    DEFAULT 0,

    max_drawdown    REAL    DEFAULT 0,
    sharpe_ratio    REAL    DEFAULT 0,
    profit_factor   REAL    DEFAULT 0,

    notes           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

SCHEMA_ERRORS = """
CREATE TABLE IF NOT EXISTS errors (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
    module          TEXT    NOT NULL,
    error_type      TEXT    NOT NULL,
    message         TEXT    NOT NULL,
    traceback       TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

# Indexes for common queries
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_signals_symbol    ON signals(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_signals_status    ON signals(status);",
    "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_trades_symbol     ON trades(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_trades_mode       ON trades(mode);",
    "CREATE INDEX IF NOT EXISTS idx_trades_status     ON trades(status);",
    "CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);",
    "CREATE INDEX IF NOT EXISTS idx_perf_date         ON performance(date);",
    "CREATE INDEX IF NOT EXISTS idx_errors_module     ON errors(module);",
]


# ╔═══════════════════════════════════════════════════════════╗
# ║                    DATABASE CLASS                        ║
# ╚═══════════════════════════════════════════════════════════╝

class Database:
    """Thread-safe SQLite database manager."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_PATH)
        self._lock = threading.Lock()
        self._create_tables()
        self._migrate_tables()
        logger.info(f"Database initialized: {self.db_path}")

    # ── Connection ───────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _execute(self, query: str, params: tuple = (),
                 fetch: str = "none") -> Any:
        """Thread-safe query execution."""
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(query, params)
                if fetch == "one":
                    result = cursor.fetchone()
                elif fetch == "all":
                    result = cursor.fetchall()
                else:
                    result = cursor.lastrowid
                conn.commit()
                return result
            except sqlite3.Error as e:
                conn.rollback()
                logger.error(f"DB error: {e} | Query: {query[:100]}")
                raise
            finally:
                conn.close()

    def _execute_many(self, query: str,
                      data: List[tuple]) -> int:
        """Execute a query for multiple rows. Returns rows affected."""
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.executemany(query, data)
                conn.commit()
                return cursor.rowcount
            except sqlite3.Error as e:
                conn.rollback()
                logger.error(f"DB executemany error: {e}")
                raise
            finally:
                conn.close()

    # ── Table Creation ───────────────────────────────────────

    def _create_tables(self):
        """Create all tables and indexes if they don't exist."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(SCHEMA_SIGNALS)
                conn.execute(SCHEMA_TRADES)
                conn.execute(SCHEMA_PERFORMANCE)
                conn.execute(SCHEMA_ERRORS)
                for idx_sql in INDEXES:
                    conn.execute(idx_sql)
                conn.commit()
                logger.debug("Database tables verified/created")
            except sqlite3.Error as e:
                logger.error(f"Failed to create tables: {e}", exc_info=True)
                raise
            finally:
                conn.close()

    # ──────────────────────────────────────────────────────────
    # Schema migration for existing databases.
    # Safe to run repeatedly (checks columns before ALTER).
    # ──────────────────────────────────────────────────────────
    def _migrate_tables(self):
        """Add columns missing from older schema versions."""
        with self._lock:
            conn = self._connect()
            try:
                migrated = 0

                # ── Trades table migrations ──
                cursor = conn.execute("PRAGMA table_info(trades)")
                trade_cols = {row[1] for row in cursor.fetchall()}

                trade_migrations = [
                    ("position_size_usd", "REAL DEFAULT 0"),
                    ("signal_type",       "TEXT"),
                    ("confidence",        "REAL DEFAULT 0"),
                    ("margin_used",       "REAL DEFAULT 0"),
                ]
                for col_name, col_type in trade_migrations:
                    if col_name not in trade_cols:
                        conn.execute(
                            f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}"
                        )
                        migrated += 1
                        logger.info(f"Migration: added trades.{col_name}")

                # ── Signals table migrations ──
                cursor = conn.execute("PRAGMA table_info(signals)")
                signal_cols = {row[1] for row in cursor.fetchall()}

                signal_migrations = [
                    ("combined_score", "REAL DEFAULT 0"),
                ]
                for col_name, col_type in signal_migrations:
                    if col_name not in signal_cols:
                        conn.execute(
                            f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}"
                        )
                        migrated += 1
                        logger.info(f"Migration: added signals.{col_name}")

                if migrated > 0:
                    conn.commit()
                    logger.info(f"Migration complete: {migrated} column(s) added")
                else:
                    logger.debug("Migration: no changes needed")

            except sqlite3.Error as e:
                logger.error(f"Migration error: {e}", exc_info=True)
            finally:
                conn.close()

    # ──────────────────────────────────────────────────────────
    #  SIGNALS CRUD
    # ──────────────────────────────────────────────────────────

    def save_signal(self, data: Dict) -> int:
        """Save a prediction signal. Returns Signal ID."""
        columns = [
            "timestamp", "symbol", "direction", "confidence",
            "combined_score",
            "entry_price", "stop_loss", "take_profit",
            "timeframe", "horizon_hours",
            "ml_direction", "ml_confidence", "ml_agreement",
            "sentiment_score", "ai_direction", "ai_confidence",
            "funding_signal", "market_signal",
            "status", "reject_reason", "notes",
        ]
        present = {k: data[k] for k in columns if k in data}

        if "symbol" not in present or "direction" not in present:
            raise ValueError("Signal must have 'symbol' and 'direction'")

        present.setdefault("timestamp", datetime.now().isoformat())
        present.setdefault("confidence", 0)
        present.setdefault("status", "active")

        cols = ", ".join(present.keys())
        placeholders = ", ".join(["?"] * len(present))
        query = f"INSERT INTO signals ({cols}) VALUES ({placeholders})"

        signal_id = self._execute(query, tuple(present.values()))

        # FIX: display confidence as percentage correctly
        conf = present.get("confidence", 0)
        score = present.get("combined_score", 0)
        logger.info(
            f"Signal saved #{signal_id}: {present.get('symbol')} "
            f"{present.get('direction')} | "
            f"conf={conf:.1%} score={score:+.3f}"
        )
        return signal_id

    def get_signals(self, symbol: str = None, status: str = None,
                    limit: int = 50) -> List[Dict]:
        """Get signals, newest first."""
        query = "SELECT * FROM signals WHERE 1=1"
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._execute(query, tuple(params), fetch="all")
        return [dict(r) for r in rows] if rows else []

    def update_signal_status(self, signal_id: int, status: str,
                             notes: str = None):
        """Update a signal's status."""
        if notes:
            self._execute(
                "UPDATE signals SET status=?, notes=? WHERE id=?",
                (status, notes, signal_id)
            )
        else:
            self._execute(
                "UPDATE signals SET status=? WHERE id=?",
                (status, signal_id)
            )
        logger.debug(f"Signal #{signal_id} → status={status}")

    # ──────────────────────────────────────────────────────────
    #  TRADES CRUD
    # ──────────────────────────────────────────────────────────

    def save_trade(self, data: Dict) -> int:
        """Save a trade record. Returns Trade ID."""
        columns = [
            "signal_id", "symbol", "direction", "mode",
            "entry_price", "entry_time", "exit_price", "exit_time",
            "quantity", "leverage", "stop_loss", "take_profit",
            "position_size_usd", "signal_type", "confidence", "margin_used",
            "pnl_usd", "pnl_percent", "commission",
            "status", "exit_reason", "notes",
        ]

        present = {k: data[k] for k in columns if k in data}

        if not all(k in present for k in ("symbol", "direction",
                                           "entry_price", "entry_time")):
            raise ValueError(
                "Trade needs: symbol, direction, entry_price, entry_time"
            )

        present.setdefault("mode", "backtest")
        present.setdefault("status", "open")
        present.setdefault("leverage", 1)

        cols = ", ".join(present.keys())
        placeholders = ", ".join(["?"] * len(present))
        query = f"INSERT INTO trades ({cols}) VALUES ({placeholders})"

        trade_id = self._execute(query, tuple(present.values()))
        logger.info(
            f"Trade saved #{trade_id}: {present['mode'].upper()} "
            f"{present['symbol']} {present['direction']} "
            f"@ {present['entry_price']}"
            f" size=${present.get('position_size_usd', 0):,.2f}"
        )
        return trade_id

    def close_trade(self, trade_id: int, exit_price: float,
                    exit_reason: str = "manual",
                    pnl_usd: float = 0, pnl_percent: float = 0,
                    commission: float = 0):
        """Close an open trade with exit details."""
        self._execute(
            """UPDATE trades
               SET exit_price=?, exit_time=?, exit_reason=?,
                   pnl_usd=?, pnl_percent=?, commission=?,
                   status='closed'
               WHERE id=?""",
            (exit_price, datetime.now().isoformat(), exit_reason,
             pnl_usd, pnl_percent, commission, trade_id)
        )
        logger.info(
            f"Trade #{trade_id} closed: {exit_reason} | "
            f"PnL: ${pnl_usd:+.2f} ({pnl_percent:+.2f}%)"
        )

    def update_trade(self, trade_id: int, updates: Dict):
        """Update arbitrary trade fields."""
        if not updates:
            return
        set_clause = ", ".join(f"{k}=?" for k in updates.keys())
        query = f"UPDATE trades SET {set_clause} WHERE id=?"
        params = list(updates.values()) + [trade_id]
        self._execute(query, tuple(params))
        logger.debug(f"Trade #{trade_id} updated: {list(updates.keys())}")

    def get_trades(self, symbol: str = None, mode: str = None,
                   status: str = None, limit: int = 100) -> List[Dict]:
        """Get trades, newest first."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._execute(query, tuple(params), fetch="all")
        return [dict(r) for r in rows] if rows else []

    def get_open_trades(self, mode: str = None) -> List[Dict]:
        """Get all currently open trades."""
        return self.get_trades(mode=mode, status="open", limit=100)

    # ──────────────────────────────────────────────────────────
    #  RISK QUERIES (used by risk_manager.py)
    # ──────────────────────────────────────────────────────────

    def get_daily_pnl(self, mode: str = "paper") -> float:
        """Sum of PnL USD for trades closed today."""
        today = datetime.now().strftime("%Y-%m-%d")
        row = self._execute(
            """SELECT COALESCE(SUM(pnl_usd), 0) as total
               FROM trades
               WHERE mode=? AND status='closed'
                 AND date(exit_time) = ?""",
            (mode, today), fetch="one"
        )
        return float(row["total"]) if row else 0.0

    def get_weekly_pnl(self, mode: str = "paper") -> float:
        """Sum of PnL USD for trades closed in last 7 days."""
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        row = self._execute(
            """SELECT COALESCE(SUM(pnl_usd), 0) as total
               FROM trades
               WHERE mode=? AND status='closed'
                 AND exit_time >= ?""",
            (mode, week_ago), fetch="one"
        )
        return float(row["total"]) if row else 0.0

    def get_consecutive_losses(self, mode: str = "paper") -> int:
        """Count consecutive losing trades from the most recent."""
        rows = self._execute(
            """SELECT pnl_usd FROM trades
               WHERE mode=? AND status='closed'
               ORDER BY id DESC LIMIT 20""",
            (mode,), fetch="all"
        )
        if not rows:
            return 0
        count = 0
        for row in rows:
            if float(row["pnl_usd"]) < 0:
                count += 1
            else:
                break
        return count

    def get_total_pnl(self, mode: str = "paper") -> float:
        """Total PnL in USD across all closed trades."""
        row = self._execute(
            """SELECT COALESCE(SUM(pnl_usd), 0) as total
               FROM trades
               WHERE mode=? AND status='closed'""",
            (mode,), fetch="one"
        )
        return float(row["total"]) if row else 0.0

    def get_open_position_count(self, mode: str = "paper") -> int:
        """Count currently open positions."""
        row = self._execute(
            """SELECT COUNT(*) as cnt FROM trades
               WHERE mode=? AND status='open'""",
            (mode,), fetch="one"
        )
        return int(row["cnt"]) if row else 0

    # ──────────────────────────────────────────────────────────
    #  PERFORMANCE SNAPSHOTS
    # ──────────────────────────────────────────────────────────

    def save_performance(self, data: Dict) -> int:
        """Save a daily performance snapshot."""
        columns = [
            "date", "mode", "symbol",
            "total_trades", "winning_trades", "losing_trades", "win_rate",
            "total_pnl", "avg_pnl", "best_trade", "worst_trade",
            "max_drawdown", "sharpe_ratio", "profit_factor", "notes",
        ]
        present = {k: data[k] for k in columns if k in data}
        present.setdefault("date", datetime.now().strftime("%Y-%m-%d"))
        present.setdefault("mode", "backtest")
        present.setdefault("symbol", "ALL")

        cols = ", ".join(present.keys())
        placeholders = ", ".join(["?"] * len(present))
        query = f"INSERT INTO performance ({cols}) VALUES ({placeholders})"
        perf_id = self._execute(query, tuple(present.values()))
        logger.debug(f"Performance snapshot saved #{perf_id}")
        return perf_id

    def get_performance(self, mode: str = None,
                        days: int = 30) -> List[Dict]:
        """Get performance snapshots for the last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        query = "SELECT * FROM performance WHERE date >= ?"
        params = [cutoff]
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        query += " ORDER BY date DESC"
        rows = self._execute(query, tuple(params), fetch="all")
        return [dict(r) for r in rows] if rows else []

    # ──────────────────────────────────────────────────────────
    #  ERROR LOGGING
    # ──────────────────────────────────────────────────────────

    def save_error(self, module: str, error_type: str,
                   message: str, traceback_str: str = None) -> int:
        """Log an error to the database."""
        error_id = self._execute(
            """INSERT INTO errors (module, error_type, message, traceback)
               VALUES (?, ?, ?, ?)""",
            (module, error_type, message, traceback_str)
        )
        logger.debug(f"Error logged #{error_id}: [{module}] {error_type}")
        return error_id

    def get_errors(self, module: str = None,
                   limit: int = 50) -> List[Dict]:
        """Get recent errors, optionally filtered by module."""
        query = "SELECT * FROM errors WHERE 1=1"
        params = []
        if module:
            query += " AND module = ?"
            params.append(module)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self._execute(query, tuple(params), fetch="all")
        return [dict(r) for r in rows] if rows else []

    # ──────────────────────────────────────────────────────────
    #  STATISTICS
    # ──────────────────────────────────────────────────────────

    def get_stats(self, mode: str = None, symbol: str = None,
                  days: int = 30) -> Dict:
        """Calculate trading statistics for a period."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        query = """
            SELECT
                COUNT(*)                           as total_trades,
                SUM(CASE WHEN pnl_usd > 0
                    THEN 1 ELSE 0 END)             as winning_trades,
                SUM(CASE WHEN pnl_usd <= 0
                    THEN 1 ELSE 0 END)             as losing_trades,
                COALESCE(SUM(pnl_percent), 0)      as total_pnl,
                COALESCE(AVG(pnl_percent), 0)      as avg_pnl,
                COALESCE(MAX(pnl_percent), 0)      as best_trade,
                COALESCE(MIN(pnl_percent), 0)      as worst_trade,
                COALESCE(SUM(pnl_usd), 0)          as total_pnl_usd,
                COALESCE(SUM(commission), 0)        as total_commission
            FROM trades
            WHERE status = 'closed' AND entry_time >= ?
        """
        params = [cutoff]
        if mode:
            query += " AND mode = ?"
            params.append(mode)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        row = self._execute(query, tuple(params), fetch="one")

        if not row or row["total_trades"] == 0:
            return {
                "total_trades": 0, "winning_trades": 0,
                "losing_trades": 0, "win_rate": 0.0,
                "total_pnl": 0.0, "avg_pnl": 0.0,
                "best_trade": 0.0, "worst_trade": 0.0,
                "profit_factor": 0.0, "total_pnl_usd": 0.0,
                "total_commission": 0.0,
            }

        total = row["total_trades"]
        wins = row["winning_trades"] or 0

        # Profit factor
        pf_row = self._execute(
            """SELECT
                COALESCE(SUM(CASE WHEN pnl_usd > 0
                    THEN pnl_usd ELSE 0 END), 0) as gross_profit,
                COALESCE(SUM(CASE WHEN pnl_usd < 0
                    THEN ABS(pnl_usd) ELSE 0 END), 0.01) as gross_loss
               FROM trades
               WHERE status='closed' AND entry_time >= ?"""
            + (" AND mode=?" if mode else "")
            + (" AND symbol=?" if symbol else ""),
            tuple(params), fetch="one"
        )

        gross_profit = pf_row["gross_profit"] if pf_row else 0
        gross_loss = pf_row["gross_loss"] if pf_row else 0.01

        return {
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": row["losing_trades"] or 0,
            "win_rate": round((wins / total * 100) if total > 0 else 0, 1),
            "total_pnl": round(row["total_pnl"], 2),
            "avg_pnl": round(row["avg_pnl"], 2),
            "best_trade": round(row["best_trade"], 2),
            "worst_trade": round(row["worst_trade"], 2),
            "profit_factor": round(
                gross_profit / max(gross_loss, 0.01), 2
            ),
            "total_pnl_usd": round(row["total_pnl_usd"], 2),
            "total_commission": round(row["total_commission"], 4),
        }

    # ──────────────────────────────────────────────────────────
    #  UTILITIES
    # ──────────────────────────────────────────────────────────

    def count_rows(self, table: str) -> int:
        """Count rows in any table."""
        allowed = ("signals", "trades", "performance", "errors")
        if table not in allowed:
            raise ValueError(f"Table must be one of {allowed}")
        row = self._execute(
            f"SELECT COUNT(*) as cnt FROM {table}", fetch="one"
        )
        return int(row["cnt"]) if row else 0

    def cleanup_old(self, days: int = 90):
        """Delete records older than N days from errors table."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        self._execute("DELETE FROM errors WHERE timestamp < ?", (cutoff,))
        logger.info(f"Cleaned up errors older than {days} days")

    def get_table_info(self) -> Dict[str, int]:
        """Return row counts for all tables."""
        return {
            "signals": self.count_rows("signals"),
            "trades": self.count_rows("trades"),
            "performance": self.count_rows("performance"),
            "errors": self.count_rows("errors"),
        }


# ╔═══════════════════════════════════════════════════════════╗
# ║                   SINGLETON ACCESS                       ║
# ╚═══════════════════════════════════════════════════════════╝

_db_instance: Optional[Database] = None
_db_lock = threading.Lock()


def get_db(db_path: str = None) -> Database:
    """Get the shared Database instance (singleton)."""
    global _db_instance
    with _db_lock:
        if _db_instance is None:
            _db_instance = Database(db_path)
        return _db_instance


# ╔═══════════════════════════════════════════════════════════╗
# ║                   STANDALONE TEST                        ║
# ╚═══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    import os

    print("=" * 60)
    print("  DATABASE TEST (v3.0.2 — combined_score)")
    print("=" * 60)

    test_db_path = Path(__file__).parent.parent / "test_agent.db"
    if test_db_path.exists():
        os.remove(test_db_path)

    db = Database(str(test_db_path))

    # ── Test: save signal WITH combined_score ──
    print("\n--- Test: Signal with combined_score ---")
    s1_id = db.save_signal({
        "symbol": "BTCUSDT",
        "direction": "SHORT",
        "confidence": 0.42,
        "combined_score": -0.274,
        "entry_price": 69000.0,
    })
    signals = db.get_signals(limit=1)
    s = signals[0]
    assert abs(float(s.get("combined_score", 0)) - (-0.274)) < 0.001, \
        f"combined_score not stored! Got: {s.get('combined_score')}"
    print(f"  ✅ Signal #{s1_id}: combined_score={s['combined_score']}")

    # ── Test: save trade WITH position_size_usd ──
    print("\n--- Test: Trade with position_size_usd ---")
    t1_id = db.save_trade({
        "symbol": "BTCUSDT",
        "direction": -1,
        "mode": "paper",
        "entry_price": 68000.0,
        "entry_time": datetime.now().isoformat(),
        "quantity": 0.074,
        "leverage": 3,
        "stop_loss": 69360.0,
        "take_profit": 65280.0,
        "position_size_usd": 5000.0,
        "signal_type": "SHORT",
        "confidence": 0.65,
        "margin_used": 1666.67,
    })
    trades = db.get_trades(limit=1)
    t = trades[0]
    assert float(t["position_size_usd"]) == 5000.0, \
        f"position_size_usd not stored! Got: {t.get('position_size_usd')}"
    print(f"  ✅ Trade #{t1_id}: position_size_usd=${t['position_size_usd']:,.2f}")

    # Close with PnL
    db.close_trade(t1_id, 67000.0, "take_profit",
                   pnl_usd=73.53, pnl_percent=0.74, commission=4.0)

    # Test get_total_pnl returns USD
    total = db.get_total_pnl("paper")
    assert abs(total - 73.53) < 0.01, f"get_total_pnl should be USD, got {total}"
    print(f"  ✅ get_total_pnl = ${total:,.2f} (USD)")

    # ── Test: migration idempotent ──
    print("\n--- Test: Migration idempotent ---")
    db._migrate_tables()
    print(f"  ✅ Re-migration safe")

    # Cleanup
    if test_db_path.exists():
        os.remove(test_db_path)

    print("\n" + "=" * 60)
    print("  ✅ All database tests passed!")
    print("=" * 60)