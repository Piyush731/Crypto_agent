"""Pre-registered deterministic multi-timeframe trend strategy (V4-T3)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .base import ExecutionPolicy, LabelSet, StrategyPlugin


@dataclass
class TrendAlignmentStrategy(StrategyPlugin):
    strategy_id: str = "trend_alignment_5m"
    version: int = 1
    description: str = (
        "Enter only on a fresh 5m momentum trigger aligned with completed "
        "15m setup and 1h trend; deterministic, no ML."
    )
    policy: ExecutionPolicy = field(
        default_factory=lambda: ExecutionPolicy(
            stop_atr_multiple=2.0,
            take_profit_atr_multiple=4.0,
            max_holding_bars=96,
            confidence_threshold=0.50,
        )
    )

    def build_labels(self, features: pd.DataFrame, bars: pd.DataFrame) -> LabelSet:
        raise NotImplementedError("Deterministic strategy does not train labels")

    def probabilities_to_signals(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Deterministic strategy does not use model probabilities")

    def features_to_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        required = {
            "tf1h_ema21_dist", "tf1h_ema21_slope3", "tf1h_adx14",
            "tf15_ema21_dist", "tf15_macd_hist",
            "ema21_dist", "macd_hist", "ret_3", "rsi14", "adx14",
        }
        missing = required - set(features.columns)
        if missing:
            raise ValueError(f"Missing trend strategy features: {sorted(missing)}")

        long_regime = (
            (features["tf1h_ema21_dist"] > 0)
            & (features["tf1h_ema21_slope3"] > 0)
            & (features["tf1h_adx14"] >= 20)
            & (features["tf15_ema21_dist"] > 0)
            & (features["tf15_macd_hist"] > 0)
        )
        short_regime = (
            (features["tf1h_ema21_dist"] < 0)
            & (features["tf1h_ema21_slope3"] < 0)
            & (features["tf1h_adx14"] >= 20)
            & (features["tf15_ema21_dist"] < 0)
            & (features["tf15_macd_hist"] < 0)
        )
        long_trigger = (
            long_regime
            & (features["ema21_dist"] > 0)
            & (features["macd_hist"] > 0)
            & (features["ret_3"] > 0)
            & features["rsi14"].between(50, 72)
            & (features["adx14"] >= 18)
        )
        short_trigger = (
            short_regime
            & (features["ema21_dist"] < 0)
            & (features["macd_hist"] < 0)
            & (features["ret_3"] < 0)
            & features["rsi14"].between(28, 50)
            & (features["adx14"] >= 18)
        )

        # One signal on transition into a setup; do not emit every 5m bar.
        fresh_long = long_trigger & ~long_trigger.shift(1, fill_value=False)
        fresh_short = short_trigger & ~short_trigger.shift(1, fill_value=False)
        direction = pd.Series(0, index=features.index, dtype="int8")
        direction[fresh_long] = 1
        direction[fresh_short] = -1
        confidence = pd.Series(0.0, index=features.index)
        confidence[direction != 0] = 0.75
        return pd.DataFrame(
            {"direction": direction, "confidence": confidence},
            index=features.index,
        )
