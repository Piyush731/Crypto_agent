import pandas as pd

from portfolio_v4.cross_sectional_momentum import CrossSectionalMomentum
from trading.portfolio_engine_v4 import PortfolioEngineV4


def test_cross_sectional_policy_is_market_neutral_sized():
    config = CrossSectionalMomentum().config
    assert config.max_notional_per_leg_pct == 25.0
    assert config.risk_per_leg_pct == 0.25
    assert config.rebalance_hours == 4


def test_signal_executes_on_first_5m_open_strictly_after_availability():
    signal_time = pd.Timestamp("2026-01-01T12:00:00Z")
    timeline = pd.date_range(signal_time, periods=3, freq="5min")
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
        },
        index=timeline,
    )
    bars = {symbol: frame.copy() for symbol in ["BTC", "ETH", "SOL", "BNB"]}
    schedule = pd.DataFrame(
        [
            {
                "long_symbol": "BTC",
                "short_symbol": None,
                "long_atr_pct": 1.0,
                "short_atr_pct": None,
            }
        ],
        index=pd.DatetimeIndex([signal_time]),
    )

    result = PortfolioEngineV4().run(
        bars=bars,
        schedule=schedule,
        strategy=CrossSectionalMomentum(),
        development_end=timeline[-1],
    )

    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert pd.Timestamp(trade["signal_time"]) == signal_time
    assert (
        pd.Timestamp(trade["entry_time"]) - pd.Timestamp(trade["signal_time"])
    ).total_seconds() == 300
    assert pd.Timestamp(trade["entry_time"]) > pd.Timestamp(trade["signal_time"])
    # Entry is based on the 12:05 open (101), never the unknowable 12:00 open.
    assert trade["entry_price"] > 101.0
