import pandas as pd

from portfolio_v4.time_series_momentum import TimeSeriesMomentum


def make_frame(multiplier: float):
    index = pd.date_range("2020-12-01", periods=2300, freq="1h", tz="UTC")
    close = pd.Series(
        [100.0 * multiplier**value for value in range(len(index))],
        index=index,
    )
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )


def test_t11_configuration_is_frozen():
    strategy = TimeSeriesMomentum()
    cfg = strategy.config
    assert len(strategy.candidate_symbols) == 9
    assert (cfg.horizon_7d_hours, cfg.horizon_30d_hours, cfg.horizon_90d_hours) == (168, 720, 2160)
    assert cfg.maximum_positions == 6
    assert cfg.risk_per_leg_pct == 0.10
    assert cfg.max_notional_per_leg_pct == 10.0
    assert cfg.max_gross_notional_pct == 60.0
    assert cfg.stop_atr_multiple == 3.0


def test_up_and_down_series_receive_opposite_two_of_three_votes():
    strategy = TimeSeriesMomentum()
    up = strategy.score_symbol(make_frame(1.0001), "UP").dropna()
    down = strategy.score_symbol(make_frame(0.9999), "DOWN").dropna()
    assert int(up.iloc[-1]["UP_direction"]) == 1
    assert int(down.iloc[-1]["DOWN_direction"]) == -1


def test_feature_availability_is_shifted_one_hour():
    strategy = TimeSeriesMomentum()
    frame = make_frame(1.0001)
    scored = strategy.score_symbol(frame, "BTC")
    assert scored.index[0].value - frame.index[0].value == 3_600_000_000_000
