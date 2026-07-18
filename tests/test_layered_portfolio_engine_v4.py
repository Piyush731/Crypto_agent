import pandas as pd

from portfolio_v4.layered_regime_trend import LayeredRegimeTrend
from trading.layered_portfolio_engine_v4 import LayeredPortfolioEngineV4


def make_schedule(signal_time, longs, shorts, atr_pct):
    return pd.DataFrame(
        [{
            "regime": "bull" if longs else "bear",
            "long_symbols": tuple(longs),
            "short_symbols": tuple(shorts),
            "atr_pct": atr_pct,
            "positive_breadth": 0.8 if longs else 0.2,
        }],
        index=pd.DatetimeIndex([signal_time]),
    )


def test_multi_leg_entries_are_causal_and_gross_capped():
    signal = pd.Timestamp("2026-01-01T12:00:00Z")
    timeline = pd.date_range(signal, periods=3, freq="5min")
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [100.5, 100.5, 100.5],
            "low": [99.5, 99.5, 99.5],
            "close": [100.0, 100.0, 100.0],
        },
        index=timeline,
    )
    bars = {symbol: frame.copy() for symbol in ["BTC", "ETH"]}
    schedule = make_schedule(signal, ["BTC", "ETH"], [], {"BTC": 2.0, "ETH": 2.0})
    result = LayeredPortfolioEngineV4().run(
        bars, schedule, LayeredRegimeTrend(), timeline[-1]
    )
    assert len(result["trades"]) == 2
    assert all(trade["entry_time"] > trade["signal_time"] for trade in result["trades"])
    assert all(trade["entry_time"] == timeline[1] for trade in result["trades"])
    assert sum(trade["notional"] for trade in result["trades"]) <= 6000.01


def test_trailing_stop_arms_then_applies_from_next_bar():
    signal = pd.Timestamp("2026-01-01T12:00:00Z")
    timeline = pd.date_range(signal, periods=3, freq="5min")
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 102.0],
            "high": [100.5, 103.5, 102.2],
            "low": [99.5, 99.5, 101.0],
            "close": [100.0, 103.0, 101.5],
        },
        index=timeline,
    )
    schedule = make_schedule(signal, ["BTC"], [], {"BTC": 1.0})
    result = LayeredPortfolioEngineV4().run(
        {"BTC": frame}, schedule, LayeredRegimeTrend(), timeline[-1]
    )
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["trailing_armed"] is True
    assert trade["exit_reason"] == "trailing_stop"
    assert trade["exit_time"] == timeline[2]
