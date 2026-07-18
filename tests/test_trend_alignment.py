import pandas as pd

from strategies_v4.trend_alignment import TrendAlignmentStrategy


def test_trend_strategy_emits_only_fresh_transition():
    index = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    features = pd.DataFrame(
        {
            "tf1h_ema21_dist": [1, 1, 1],
            "tf1h_ema21_slope3": [1, 1, 1],
            "tf1h_adx14": [25, 25, 25],
            "tf15_ema21_dist": [1, 1, 1],
            "tf15_macd_hist": [1, 1, 1],
            "ema21_dist": [1, 1, 1],
            "macd_hist": [1, 1, 1],
            "ret_3": [1, 1, 1],
            "rsi14": [60, 60, 60],
            "adx14": [25, 25, 25],
        },
        index=index,
    )
    signals = TrendAlignmentStrategy().features_to_signals(features)
    assert signals["direction"].tolist() == [1, 0, 0]
