from portfolio_v4.registry import PORTFOLIO_STRATEGIES, get_portfolio_strategy


def test_portfolio_trials_are_pre_registered():
    assert set(PORTFOLIO_STRATEGIES) == {
        "cross_sectional_momentum_4h_v1",
        "cross_sectional_momentum_8h_v1",
        "cost_aware_dual_trend_8h_v1",
    }
    assert (
        get_portfolio_strategy("cross_sectional_momentum_8h_v1")
        .config.rebalance_hours
        == 8
    )
    t9 = get_portfolio_strategy("cost_aware_dual_trend_8h_v1")
    assert t9.config.long_lookback_hours == 720
    assert len(t9.candidate_symbols) == 10
