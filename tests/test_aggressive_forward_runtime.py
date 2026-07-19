import pytest

from paper_v4.aggressive_forward_runner import validate_environment
from portfolio_v4.aggressive_forward_tsm import AggressiveForwardTimeSeriesMomentum


def test_aggressive_forward_configuration_is_frozen():
    strategy = AggressiveForwardTimeSeriesMomentum()
    assert strategy.forward_only is True
    assert strategy.hard_halt_drawdown_pct == 25.0
    assert strategy.config.risk_per_leg_pct == 0.20
    assert strategy.config.maximum_positions == 6
    assert strategy.config.max_notional_per_leg_pct == 15.0
    assert strategy.config.max_gross_notional_pct == 75.0


def test_aggressive_runtime_refuses_real_orders(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "paper")
    monkeypatch.setenv("MARKET_PROVIDER", "okx")
    monkeypatch.setenv("ALLOW_REAL_ORDERS", "true")
    monkeypatch.setenv("FORWARD_ONLY", "true")
    with pytest.raises(RuntimeError, match="ALLOW_REAL_ORDERS"):
        validate_environment()


def test_aggressive_runtime_requires_forward_only(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "paper")
    monkeypatch.setenv("MARKET_PROVIDER", "okx")
    monkeypatch.setenv("ALLOW_REAL_ORDERS", "false")
    monkeypatch.setenv("FORWARD_ONLY", "false")
    with pytest.raises(RuntimeError, match="FORWARD_ONLY"):
        validate_environment()
