import pandas as pd

from portfolio_v4.cross_sectional_momentum import CrossSectionalMomentum


def make_hourly(multiplier: float):
    index = pd.date_range("2025-01-01", periods=300, freq="1h", tz="UTC")
    close = pd.Series(
        [100 * (1 + multiplier) ** value for value in range(300)], index=index
    )
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 100.0,
            "confirmed": True,
        },
        index=index,
    )


def test_ranking_selects_strongest_and_weakest():
    strategy = CrossSectionalMomentum()
    schedule = strategy.build_schedule(
        {
            "BTC": make_hourly(0.0005),
            "ETH": make_hourly(0.0002),
            "SOL": make_hourly(-0.0004),
            "BNB": make_hourly(-0.0001),
        }
    )
    assert not schedule.empty
    row = schedule.iloc[-1]
    assert row["long_symbol"] == "BTC"
    assert row["short_symbol"] == "SOL"
