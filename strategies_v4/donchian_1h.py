"""Low-turnover completed-1h Donchian trend strategy (V4-T5)."""

from dataclasses import dataclass, field

import pandas as pd

from .base import ExecutionPolicy, LabelSet, StrategyPlugin


@dataclass
class Donchian1hStrategy(StrategyPlugin):
    strategy_id: str = "donchian_1h"
    version: int = 1
    description: str = (
        "Fresh completed-1h 55-bar breakout with ADX, rising/falling EMA21, "
        "and volume confirmation; designed for low turnover."
    )
    policy: ExecutionPolicy = field(
        default_factory=lambda: ExecutionPolicy(
            stop_atr_multiple=2.0,
            take_profit_atr_multiple=6.0,
            max_holding_bars=576,
            confidence_threshold=0.50,
        )
    )

    def build_labels(self, features: pd.DataFrame, bars: pd.DataFrame) -> LabelSet:
        raise NotImplementedError

    def probabilities_to_signals(self, probabilities: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def features_to_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        required = {
            "tf1h_breakout_high55_pct", "tf1h_breakout_low55_pct",
            "tf1h_volume_ratio20", "tf1h_atr_pct", "tf1h_adx14",
            "tf1h_ema21_slope3",
        }
        missing = required - set(features.columns)
        if missing:
            raise ValueError(f"Missing 1h Donchian features: {sorted(missing)}")

        long_condition = (
            (features["tf1h_breakout_high55_pct"] > 0)
            & (features["tf1h_volume_ratio20"] >= 1.0)
            & (features["tf1h_adx14"] >= 20)
            & (features["tf1h_ema21_slope3"] > 0)
        )
        short_condition = (
            (features["tf1h_breakout_low55_pct"] < 0)
            & (features["tf1h_volume_ratio20"] >= 1.0)
            & (features["tf1h_adx14"] >= 20)
            & (features["tf1h_ema21_slope3"] < 0)
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
                "atr_pct": features["tf1h_atr_pct"],
            },
            index=features.index,
        )
