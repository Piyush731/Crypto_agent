"""Completed-15m range mean-reversion strategy (V4-T6)."""

from dataclasses import dataclass, field

import pandas as pd

from .base import ExecutionPolicy, LabelSet, StrategyPlugin


@dataclass
class RangeReversion15mStrategy(StrategyPlugin):
    strategy_id: str = "range_reversion_15m"
    version: int = 1
    description: str = (
        "Fade completed-15m Bollinger extremes only while completed-1h ADX "
        "indicates a non-trending regime."
    )
    policy: ExecutionPolicy = field(
        default_factory=lambda: ExecutionPolicy(
            stop_atr_multiple=2.0,
            take_profit_atr_multiple=2.0,
            max_holding_bars=48,
            confidence_threshold=0.50,
        )
    )

    def build_labels(self, features: pd.DataFrame, bars: pd.DataFrame) -> LabelSet:
        raise NotImplementedError

    def probabilities_to_signals(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def features_to_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        required = {
            "tf15_bb_pct_b", "tf15_rsi14", "tf15_atr_pct",
            "tf15_adx14", "tf1h_adx14",
        }
        missing = required - set(features.columns)
        if missing:
            raise ValueError(f"Missing range strategy features: {sorted(missing)}")

        range_regime = (features["tf1h_adx14"] < 20) & (features["tf15_adx14"] < 22)
        long_condition = (
            range_regime
            & (features["tf15_bb_pct_b"] < 0.0)
            & (features["tf15_rsi14"] < 30)
        )
        short_condition = (
            range_regime
            & (features["tf15_bb_pct_b"] > 1.0)
            & (features["tf15_rsi14"] > 70)
        )
        fresh_long = long_condition & ~long_condition.shift(1, fill_value=False)
        fresh_short = short_condition & ~short_condition.shift(1, fill_value=False)
        direction = pd.Series(0, index=features.index, dtype="int8")
        direction[fresh_long] = 1
        direction[fresh_short] = -1
        confidence = pd.Series(0.0, index=features.index)
        confidence[direction != 0] = 0.75
        return pd.DataFrame(
            {
                "direction": direction,
                "confidence": confidence,
                "atr_pct": features["tf15_atr_pct"],
            },
            index=features.index,
        )
