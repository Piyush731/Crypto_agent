import pandas as pd

from trading.v4_backtester import V4Backtester, V4ExecutionConfig


def test_signal_executes_at_next_bar_timestamp_and_costs_are_charged():
    index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100, 100, 101, 102],
            "high": [101, 102, 103, 103],
            "low": [99, 99.5, 100, 101],
            "close": [100, 101, 102, 102],
        },
        index=index,
    )
    features = pd.DataFrame({"atr_pct": [1.0] * 4}, index=index)
    predictions = pd.DataFrame(
        {
            "prediction": [1, 0, 0, 0],
            "confidence": [0.9, 0.1, 0.1, 0.1],
        },
        index=index,
    )
    config = V4ExecutionConfig(
        take_profit_atr_multiple=1.0,
        stop_atr_multiple=1.0,
        confidence_threshold=0.45,
    )
    result = V4Backtester(config).run(bars, features, predictions)
    assert result["trades"]
    trade = result["trades"][0]
    assert trade["entry_time"] == index[0]
    assert trade["entry_price"] > bars.iloc[0]["open"]
    assert trade["entry_fee"] > 0
    assert trade["exit_fee"] > 0


def test_low_confidence_signal_does_not_trade():
    index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        index=index,
    )
    features = pd.DataFrame({"atr_pct": 1.0}, index=index)
    predictions = pd.DataFrame(
        {"prediction": 1, "confidence": 0.40}, index=index
    )
    result = V4Backtester().run(bars, features, predictions)
    assert result["metrics"]["trades"] == 0
