"""15m Donchian breakout with completed 1h trend filter (V4-T4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .base import ExecutionPolicy, LabelSet, StrategyPlugin


@dataclass
class Donchian15mStrategy(StrategyPlugin):
    strategy_id: str = "donchian_15m"
    version: int = 1
    description: str = (
        "Trade fresh completed-15m 20-bar breakouts only when completed 1h "
        "EMA trend and ADX agree; deterministic, lower-turnover hypothesis."
    )
    policy: ExecutionPolicy = field(
        default_factory=lambda: ExecutionPolicy(
            stop_atr_multiple=2.0,
            take_profit_atr_multiple=4.0,
            max_holding_bars=192,
            confidence_threshold=0.50,
        )
    )

    def build_labels(self, features: pd.DataFrame, bars: pd.DataFrame) -> LabelSet:
        raise NotImplementedError("Deterministic strategy does not train labels")

    def probabilities_to_signals(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Deterministic strategy does not use probabilities")

    def features_to_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        required = {
            "tf15_breakout_high20_pct",
            "tf15_breakout_low20_pct",
            "tf15_volume_ratio20",
            "tf15_atr_pct",
            "tf15_adx14",
            "tf1h_ema50_dist",
            "tf1h_ema21_slope3",
            "tf1h_adx14",
        }
        missing = required - set(features.columns)
        if missing:
            raise ValueError(f"Missing Donchian features: {sorted(missing)}")

        long_condition = (
            (features["tf15_breakout_high20_pct"] > 0)
            & (features["tf15_volume_ratio20"] >= 1.0)
            & (features["tf15_adx14"] >= 18)
            & (features["tf1h_ema50_dist"] > 0)
            & (features["tf1h_ema21_slope3"] > 0)
            & (features["tf1h_adx14"] >= 20)
        )
        short_condition = (
            (features["tf15_breakout_low20_pct"] < 0)
            & (features["tf15_volume_ratio20"] >= 1.0)
            & (features["tf15_adx14"] >= 18)
            & (features["tf1h_ema50_dist"] < 0)
            & (features["tf1h_ema21_slope3"] < 0)
            & (features["tf1h_adx14"] >= 20)
        )

        # 15m features are constant for three 5m decision rows. Transition
        # gating emits only once when a newly completed 15m breakout appears.
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
