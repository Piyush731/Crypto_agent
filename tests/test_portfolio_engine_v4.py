from portfolio_v4.cross_sectional_momentum import CrossSectionalMomentum


def test_cross_sectional_policy_is_market_neutral_sized():
    config = CrossSectionalMomentum().config
    assert config.max_notional_per_leg_pct == 25.0
    assert config.risk_per_leg_pct == 0.25
    assert config.rebalance_hours == 4
