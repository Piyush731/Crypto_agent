import pandas as pd

from portfolio_v4.time_series_momentum import TimeSeriesMomentum
from trading.time_series_portfolio_engine_v4 import TimeSeriesPortfolioEngineV4


def test_t11_entries_use_next_5m_open_and_respect_leg_cap():
    signal = pd.Timestamp("2022-01-03T00:00:00Z")
    timeline = pd.date_range(signal, periods=3, freq="5min")
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [100.5, 101.5, 102.5],
            "low": [99.5, 100.5, 101.5],
            "close": [100.0, 101.0, 102.0],
        },
        index=timeline,
    )
    schedule = pd.DataFrame(
        [{
            "long_symbols": ("BTC",),
            "short_symbols": (),
            "atr_pct": {"BTC": 1.0},
            "signed_score": {"BTC": 1.5},
        }],
        index=pd.DatetimeIndex([signal]),
    )
    result = TimeSeriesPortfolioEngineV4().run(
        {"BTC": frame}, schedule, TimeSeriesMomentum(), timeline[-1]
    )
    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["entry_time"] == timeline[1]
    assert trade["entry_time"] > trade["signal_time"]
    assert trade["notional"] <= 1000.01
