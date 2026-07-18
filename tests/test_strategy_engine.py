import pandas as pd

from strategies_v4.triple_barrier import TripleBarrierStrategy
from trading.strategy_engine import StrategyBacktestEngine


def test_strategy_engine_accepts_standard_probability_frame():
    index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100, 100, 101, 102],
            "high": [101, 104, 103, 103],
            "low": [99.5, 99.5, 100, 101],
            "close": [100, 103, 102, 102],
        },
        index=index,
    )
    features = pd.DataFrame({"atr_pct": 1.0}, index=index)
    probabilities = pd.DataFrame(
        {
            "prob_short": [0.05, 0.33, 0.33, 0.33],
            "prob_hold": [0.05, 0.34, 0.34, 0.34],
            "prob_long": [0.90, 0.33, 0.33, 0.33],
        },
        index=index,
    )
    result = StrategyBacktestEngine().run(
        bars, features, probabilities, TripleBarrierStrategy()
    )
    assert result["strategy"]["strategy_id"] == "triple_barrier_5m"
    assert result["metrics"]["trades"] >= 1
