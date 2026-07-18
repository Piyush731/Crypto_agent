import pandas as pd

from portfolio_v4.cost_aware_dual_trend import CostAwareDualTrend


def test_t9_configuration_is_frozen():
    strategy = CostAwareDualTrend()
    cfg = strategy.config
    assert len(strategy.candidate_symbols) == 10
    assert cfg.return_72h_weight == 0.40
    assert cfg.return_30d_weight == 0.60
    assert cfg.long_lookback_hours == 720
    assert cfg.rebalance_hours == 8
    assert cfg.retention_rank == 3
    assert cfg.risk_per_leg_pct == 0.25
    assert cfg.max_notional_per_leg_pct == 25.0


def test_rank_buffer_retains_valid_incumbents():
    strategy = CostAwareDualTrend()
    scores = {symbol: float(10 - rank) for rank, symbol in enumerate("ABCDEFGHIJ")}
    raw = {symbol: score - 4.5 for symbol, score in scores.items()}

    long_symbol, short_symbol = strategy.select_with_buffer(
        scores=scores,
        raw=raw,
        incumbent_long="B",   # rank 2, still positive
        incumbent_short="I",  # bottom 2, still negative
    )

    assert long_symbol == "B"
    assert short_symbol == "I"


def test_rank_buffer_replaces_invalid_incumbents():
    strategy = CostAwareDualTrend()
    scores = {symbol: float(10 - rank) for rank, symbol in enumerate("ABCDEFGHIJ")}
    raw = {symbol: score - 4.5 for symbol, score in scores.items()}

    long_symbol, short_symbol = strategy.select_with_buffer(
        scores=scores,
        raw=raw,
        incumbent_long="D",   # outside top 3
        incumbent_short="G",  # outside bottom 3
    )

    assert long_symbol == "A"
    assert short_symbol == "J"


def test_completed_hourly_input_is_shifted_to_availability_time():
    strategy = CostAwareDualTrend()
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
    assert (
        scored.index[0].value - index[0].value
    ) == 3_600_000_000_000
