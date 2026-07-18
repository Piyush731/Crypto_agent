import sqlite3

import pytest

from paper_v4.runner import validate_safety_environment
from paper_v4.store import PaperStore


def test_safety_environment_rejects_non_paper(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "live")
    monkeypatch.setenv("MARKET_PROVIDER", "okx")
    monkeypatch.setenv("ALLOW_REAL_ORDERS", "false")
    with pytest.raises(RuntimeError, match="AGENT_MODE"):
        validate_safety_environment()


def test_safety_environment_requires_real_orders_false(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "paper")
    monkeypatch.setenv("MARKET_PROVIDER", "okx")
    monkeypatch.setenv("ALLOW_REAL_ORDERS", "true")
    with pytest.raises(RuntimeError, match="ALLOW_REAL_ORDERS"):
        validate_safety_environment()


def test_paper_store_is_separate_restart_safe_ledger(tmp_path):
    store = PaperStore(tmp_path / "paper.db")
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        store.set_state(connection, "capital", 10_000.0)
        store.queue(connection, "once", "hello")
        store.queue(connection, "once", "duplicate")

    reopened = PaperStore(tmp_path / "paper.db")
    assert reopened.get_state("capital") == 10_000.0
    assert len(reopened.pending_notifications()) == 1
    with sqlite3.connect(reopened.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 0
