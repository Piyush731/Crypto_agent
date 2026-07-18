import pandas as pd

from strategies_v4.triple_barrier import TripleBarrierStrategy


def test_triple_barrier_labels_clear_long_path():
    index = pd.date_range("2026-01-01", periods=60, freq="5min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": [100.2] + [104.0] * 59,
            "low": 99.5,
            "close": 103.0,
        },
        index=index,
    )
    features = pd.DataFrame({"atr_pct": 1.0}, index=index)
    result = TripleBarrierStrategy().build_labels(features, bars)
    assert result.target.iloc[0] == 1
    assert result.target_end_time.iloc[0] > result.target.index[0]


def test_low_confidence_becomes_hold():
    index = pd.DatetimeIndex([pd.Timestamp("2026-01-01T00:00:00Z")])
    probabilities = pd.DataFrame(
        {"prob_short": [0.40], "prob_hold": [0.20], "prob_long": [0.40]},
        index=index,
    )
    signals = TripleBarrierStrategy().probabilities_to_signals(probabilities)
    assert signals.iloc[0]["direction"] == 0
