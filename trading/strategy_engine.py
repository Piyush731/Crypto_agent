"""Adapter from a pluggable strategy signal frame to the common v4 engine."""

from __future__ import annotations

import pandas as pd

from strategies_v4.base import StrategyPlugin
from trading.v4_backtester import V4Backtester, V4ExecutionConfig


class StrategyBacktestEngine:
    def run_signals(
        self,
        bars: pd.DataFrame,
        features: pd.DataFrame,
        signals: pd.DataFrame,
        strategy: StrategyPlugin,
        starting_capital: float = 10_000.0,
    ) -> dict:
        required = {"direction", "confidence"}
        missing = required - set(signals.columns)
        if missing:
            raise ValueError(f"Missing standard signal columns: {sorted(missing)}")
        predictions = signals.rename(columns={"direction": "prediction"})
        policy = strategy.policy
        config = V4ExecutionConfig(
            starting_capital=starting_capital,
            risk_per_trade_pct=policy.risk_per_trade_pct,
            max_notional_pct=policy.max_notional_pct,
            max_leverage=policy.max_leverage,
            stop_atr_multiple=policy.stop_atr_multiple,
            take_profit_atr_multiple=policy.take_profit_atr_multiple,
            max_holding_bars=policy.max_holding_bars,
            fee_bps_per_side=policy.fee_bps_per_side,
            slippage_bps_per_side=policy.slippage_bps_per_side,
            half_spread_bps_per_side=policy.half_spread_bps_per_side,
            conservative_funding_bps_per_8h=policy.funding_bps_per_8h,
            confidence_threshold=policy.confidence_threshold,
        )
        result = V4Backtester(config).run(bars, features, predictions)
        result["strategy"] = strategy.metadata()
        return result

    def run(
        self,
        bars: pd.DataFrame,
        features: pd.DataFrame,
        probabilities: pd.DataFrame,
        strategy: StrategyPlugin,
        starting_capital: float = 10_000.0,
    ) -> dict:
        signals = strategy.probabilities_to_signals(probabilities)
        predictions = signals.rename(columns={"direction": "prediction"})
        policy = strategy.policy
        config = V4ExecutionConfig(
            starting_capital=starting_capital,
            risk_per_trade_pct=policy.risk_per_trade_pct,
            max_notional_pct=policy.max_notional_pct,
            max_leverage=policy.max_leverage,
            stop_atr_multiple=policy.stop_atr_multiple,
            take_profit_atr_multiple=policy.take_profit_atr_multiple,
            max_holding_bars=policy.max_holding_bars,
            fee_bps_per_side=policy.fee_bps_per_side,
            slippage_bps_per_side=policy.slippage_bps_per_side,
            half_spread_bps_per_side=policy.half_spread_bps_per_side,
            conservative_funding_bps_per_8h=policy.funding_bps_per_8h,
            confidence_threshold=policy.confidence_threshold,
        )
        result = V4Backtester(config).run(bars, features, predictions)
        result["strategy"] = strategy.metadata()
        return result
