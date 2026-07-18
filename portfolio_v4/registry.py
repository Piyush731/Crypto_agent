"""Explicit registry for shared-capital strategy trials."""

from portfolio_v4.cost_aware_dual_trend import CostAwareDualTrend
from portfolio_v4.cross_sectional_momentum import CrossSectionalMomentum
from portfolio_v4.layered_regime_trend import LayeredRegimeTrend
from portfolio_v4.cross_sectional_momentum_8h import CrossSectionalMomentum8h

PORTFOLIO_STRATEGIES = {
    "cross_sectional_momentum_4h_v1": CrossSectionalMomentum,
    "cross_sectional_momentum_8h_v1": CrossSectionalMomentum8h,
    "cost_aware_dual_trend_8h_v1": CostAwareDualTrend,
    "layered_regime_trend_12h_v1": LayeredRegimeTrend,
}


def get_portfolio_strategy(key: str):
    factory = PORTFOLIO_STRATEGIES.get(key)
    if factory is None:
        raise KeyError(
            f"Unknown portfolio strategy {key}; "
            f"available={sorted(PORTFOLIO_STRATEGIES)}"
        )
    return factory()
