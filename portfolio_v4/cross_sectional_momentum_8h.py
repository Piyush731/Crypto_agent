"""Single pre-registered turnover-reduction variant (V4-T8)."""

from portfolio_v4.cross_sectional_momentum import (
    CrossSectionalMomentum,
    CrossSectionalMomentumConfig,
)


class CrossSectionalMomentum8h(CrossSectionalMomentum):
    strategy_id = "cross_sectional_momentum_1h"
    version = 2

    def __init__(self):
        super().__init__(
            CrossSectionalMomentumConfig(
                rebalance_hours=8,
            )
        )

    def metadata(self):
        metadata = super().metadata()
        metadata["version"] = self.version
        metadata["description"] = (
            "Turnover-reduction variant: same volatility-adjusted 24h/7d "
            "cross-sectional momentum, rebalanced every 8h instead of 4h."
        )
        return metadata
