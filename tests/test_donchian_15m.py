import pandas as pd

from strategies_v4.donchian_15m import Donchian15mStrategy


def test_donchian_emits_one_fresh_breakout():
    index = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "tf15_breakout_high20_pct": [-0.1, 0.2, 0.2, 0.2],
            "tf15_breakout_low20_pct": [1.0, 1.0, 1.0, 1.0],
            "tf15_volume_ratio20": [1.2] * 4,
            "tf15_atr_pct": [0.5] * 4,
            "tf15_adx14": [25] * 4,
            "tf1h_ema50_dist": [1.0] * 4,
            "tf1h_ema21_slope3": [1.0] * 4,
            "tf1h_adx14": [25] * 4,
        },
        index=index,
    )
    signals = Donchian15mStrategy().features_to_signals(frame)
    assert signals["direction"].tolist() == [0, 1, 0, 0]
    assert (signals["atr_pct"] == 0.5).all()
