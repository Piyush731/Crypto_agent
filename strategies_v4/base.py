"""Contracts shared by every v4 strategy plugin."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExecutionPolicy:
    risk_per_trade_pct: float = 0.25
    max_notional_pct: float = 50.0
    max_leverage: float = 2.0
    stop_atr_multiple: float = 2.0
    take_profit_atr_multiple: float = 3.0
    max_holding_bars: int = 48
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 2.0
    half_spread_bps_per_side: float = 1.0
    funding_bps_per_8h: float = 1.0
    confidence_threshold: float = 0.50


@dataclass
class LabelSet:
    target: pd.Series
    target_end_time: pd.Series
    metadata: dict[str, Any]


class StrategyPlugin(ABC):
    """A strategy owns labels, signal selection, and execution policy.

    The common engine owns fills, fees, P&L, and portfolio accounting.
    """

    strategy_id: str
    version: int
    description: str
    policy: ExecutionPolicy

    @abstractmethod
    def build_labels(
        self,
        features: pd.DataFrame,
        bars: pd.DataFrame,
    ) -> LabelSet:
        raise NotImplementedError

    @abstractmethod
    def probabilities_to_signals(
        self,
        probabilities: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return columns `direction` (-1/0/1) and `confidence`."""
        raise NotImplementedError

    def features_to_signals(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Optional deterministic signal path for rule-based strategies."""
        return None

    def metadata(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "description": self.description,
            "policy": asdict(self.policy),
        }
