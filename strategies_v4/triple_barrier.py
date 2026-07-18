"""Economically aligned 5m triple-barrier ML strategy (Trial V4-T2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import ExecutionPolicy, LabelSet, StrategyPlugin


@dataclass
class TripleBarrierStrategy(StrategyPlugin):
    strategy_id: str = "triple_barrier_5m"
    version: int = 1
    description: str = (
        "Predict whether LONG or SHORT reaches 3 ATR before 2 ATR stop "
        "within 48 completed 5m bars; ambiguous/no-resolution events are HOLD."
    )
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)

    def build_labels(
        self,
        features: pd.DataFrame,
        bars: pd.DataFrame,
    ) -> LabelSet:
        bars = bars.sort_index()
        features = features.sort_index()
        horizon = self.policy.max_holding_bars
        labels = pd.Series(np.nan, index=features.index, dtype="float64")
        end_times = pd.Series(pd.NaT, index=features.index, dtype="datetime64[ns, UTC]")

        bar_position = {timestamp: position for position, timestamp in enumerate(bars.index)}
        resolved = {"long": 0, "short": 0, "hold": 0, "insufficient": 0}

        for decision_time in features.index:
            start_position = bar_position.get(decision_time)
            if start_position is None or start_position + horizon > len(bars):
                resolved["insufficient"] += 1
                continue

            entry_bar = bars.iloc[start_position]
            entry = float(entry_bar["open"])
            atr_pct = float(features.loc[decision_time, "atr_pct"])
            atr_value = entry * atr_pct / 100
            if not np.isfinite(atr_value) or atr_value <= 0:
                resolved["insufficient"] += 1
                continue

            long_stop = entry - self.policy.stop_atr_multiple * atr_value
            long_target = entry + self.policy.take_profit_atr_multiple * atr_value
            short_stop = entry + self.policy.stop_atr_multiple * atr_value
            short_target = entry - self.policy.take_profit_atr_multiple * atr_value

            long_outcome = 0
            short_outcome = 0
            path = bars.iloc[start_position : start_position + horizon]

            for _, bar in path.iterrows():
                # Conservative ordering: stop is assumed first when both touch.
                if long_outcome == 0:
                    if float(bar["low"]) <= long_stop:
                        long_outcome = -1
                    elif float(bar["high"]) >= long_target:
                        long_outcome = 1

                if short_outcome == 0:
                    if float(bar["high"]) >= short_stop:
                        short_outcome = -1
                    elif float(bar["low"]) <= short_target:
                        short_outcome = 1

                if long_outcome != 0 and short_outcome != 0:
                    break

            if long_outcome == 1 and short_outcome != 1:
                label = 1
                resolved["long"] += 1
            elif short_outcome == 1 and long_outcome != 1:
                label = -1
                resolved["short"] += 1
            else:
                label = 0
                resolved["hold"] += 1

            labels.loc[decision_time] = label
            final_bar_open = path.index[-1]
            end_times.loc[decision_time] = final_bar_open + pd.to_timedelta(5, unit="min")

        valid = labels.notna() & end_times.notna()
        return LabelSet(
            target=labels.loc[valid].astype("int8"),
            target_end_time=end_times.loc[valid],
            metadata={
                "strategy": self.metadata(),
                "events": resolved,
                "valid_events": int(valid.sum()),
            },
        )

    def probabilities_to_signals(
        self,
        probabilities: pd.DataFrame,
    ) -> pd.DataFrame:
        required = {"prob_short", "prob_hold", "prob_long"}
        missing = required - set(probabilities.columns)
        if missing:
            raise ValueError(f"Missing probability columns: {sorted(missing)}")

        ordered = probabilities[["prob_short", "prob_hold", "prob_long"]]
        confidence = ordered.max(axis=1)
        class_values = np.array([-1, 0, 1], dtype=int)
        direction = pd.Series(
            class_values[ordered.to_numpy().argmax(axis=1)],
            index=ordered.index,
            dtype="int8",
        )
        direction[(confidence < self.policy.confidence_threshold)] = 0

        return pd.DataFrame(
            {
                "direction": direction,
                "confidence": confidence.astype(float),
            },
            index=ordered.index,
        )
