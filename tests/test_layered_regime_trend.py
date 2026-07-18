import pandas as pd

from portfolio_v4.layered_regime_trend import LayeredRegimeTrend


def test_t10_configuration_is_frozen():
    strategy = LayeredRegimeTrend()
    cfg = strategy.config
    assert len(strategy.candidate_symbols) == 10
    assert cfg.return_7d_weight == 0.50
    assert cfg.return_30d_weight == 0.50
    assert cfg.breadth_threshold == 0.60
    assert cfg.rebalance_hours == 12
    assert cfg.retention_rank == 4
    assert cfg.positions_per_regime == 2
    assert cfg.risk_per_leg_pct == 0.50
    assert cfg.max_gross_notional_pct == 60.0
    assert cfg.half_risk_drawdown_pct == 7.5
    assert cfg.stop_new_drawdown_pct == 12.5


def test_bull_retains_top_four_incumbent_and_adds_best():
    strategy = LayeredRegimeTrend()
    scores = {symbol: float(10 - rank) for rank, symbol in enumerate("ABCDEFGHIJ")}
    raw = {symbol: score - 4.5 for symbol, score in scores.items()}
    longs, shorts = strategy.select_with_retention(
        scores, raw, "bull", ("D",), ()
    )
    assert longs == ("D", "A")
    assert shorts == ()


def test_bear_retains_bottom_four_incumbent_and_adds_worst():
    strategy = LayeredRegimeTrend()
    scores = {symbol: float(10 - rank) for rank, symbol in enumerate("ABCDEFGHIJ")}
    raw = {symbol: score - 4.5 for symbol, score in scores.items()}
    longs, shorts = strategy.select_with_retention(
        scores, raw, "bear", (), ("G",)
    )
    assert longs == ()
    assert shorts == ("G", "J")


def test_mixed_regime_stays_in_cash():
    strategy = LayeredRegimeTrend()
    scores = {symbol: float(10 - rank) for rank, symbol in enumerate("ABCDEFGHIJ")}
    raw = {symbol: score - 4.5 for symbol, score in scores.items()}
    assert strategy.select_with_retention(scores, raw, "mixed", ("A",), ("J",)) == ((), ())


def test_hourly_feature_availability_is_shifted_exactly_one_hour():
    strategy = LayeredRegimeTrend()
    index = pd.date_range("2025-01-01", periods=730, freq="1h", tz="UTC")
    close = pd.Series(range(100, 830), index=index, dtype=float)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )
    scored = strategy.score_symbol(frame, "BTC")
    assert scored.index[0].value - index[0].value == 3_600_000_000_000
