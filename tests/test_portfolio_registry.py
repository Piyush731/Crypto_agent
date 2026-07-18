from portfolio_v4.registry import PORTFOLIO_STRATEGIES, get_portfolio_strategy


def test_portfolio_trials_are_pre_registered():
    assert set(PORTFOLIO_STRATEGIES) == {
        "cross_sectional_momentum_4h_v1",
        "cross_sectional_momentum_8h_v1",
        "cost_aware_dual_trend_8h_v1",
        "layered_regime_trend_12h_v1",
    }
    assert (
        get_portfolio_strategy("cross_sectional_momentum_8h_v1")
        .config.rebalance_hours
        == 8
    )
    t9 = get_portfolio_strategy("cost_aware_dual_trend_8h_v1")
    assert t9.config.long_lookback_hours == 720
    assert len(t9.candidate_symbols) == 10
    t10 = get_portfolio_strategy("layered_regime_trend_12h_v1")
    assert t10.config.rebalance_hours == 12
    assert t10.config.risk_per_leg_pct == 0.50
    assert t10.engine_type == "layered_multi_leg_v1"
