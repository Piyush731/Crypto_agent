"""Frozen forward-only aggressive-paper configuration; not a validated trial."""

from portfolio_v4.time_series_momentum import (
    TimeSeriesMomentum,
    TimeSeriesMomentumConfig,
)


class AggressiveForwardTimeSeriesMomentum(TimeSeriesMomentum):
    strategy_id = "aggressive_forward_time_series_momentum_1h"
    version = 1
    forward_only = True
    hard_halt_drawdown_pct = 25.0

    def __init__(self):
        super().__init__(
            TimeSeriesMomentumConfig(
                risk_per_leg_pct=0.20,
                max_notional_per_leg_pct=15.0,
                max_gross_notional_pct=75.0,
            )
        )

    def metadata(self):
        metadata = super().metadata()
        metadata.update(
            {
                "strategy_id": self.strategy_id,
                "version": self.version,
                "forward_only": True,
                "hard_halt_drawdown_pct": self.hard_halt_drawdown_pct,
                "description": (
                    "Forward-only high-risk experimental paper configuration. "
                    "Frozen T11 signals with 0.20% risk per leg, max six legs, "
                    "75% gross cap and permanent halt at 25% drawdown."
                ),
            }
        )
        return metadata
