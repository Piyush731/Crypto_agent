"""Development-only event-driven backtester for purged OOS v4 predictions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class V4ExecutionConfig:
    starting_capital: float = 10_000.0
    risk_per_trade_pct: float = 0.25
    max_notional_pct: float = 50.0
    max_leverage: float = 2.0
    stop_atr_multiple: float = 2.0
    take_profit_atr_multiple: float = 3.0
    max_holding_bars: int = 48
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 2.0
    half_spread_bps_per_side: float = 1.0
    conservative_funding_bps_per_8h: float = 1.0
    confidence_threshold: float = 0.45


class V4Backtester:
    def __init__(self, config: V4ExecutionConfig | None = None):
        self.config = config or V4ExecutionConfig()

    def run(
        self,
        bars: pd.DataFrame,
        features: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> dict[str, Any]:
        cfg = self.config
        capital = cfg.starting_capital
        position = None
        trades = []
        equity_rows = []
        friction = (cfg.slippage_bps_per_side + cfg.half_spread_bps_per_side) / 10_000
        fee_rate = cfg.fee_bps_per_side / 10_000

        bars = bars.sort_index()
        predictions = predictions.sort_index()
        timeline = bars.index.intersection(predictions.index)
        if timeline.empty:
            raise ValueError("No overlap between OOS predictions and execution bars")

        for timestamp in timeline:
            bar = bars.loc[timestamp]
            prediction = predictions.loc[timestamp]
            if isinstance(prediction, pd.DataFrame):
                prediction = prediction.iloc[-1]

            # Signal at the previous 5m close is available at this timestamp,
            # so execution at this bar's open is causal.
            if position is None:
                direction = int(prediction["prediction"])
                confidence = float(prediction["confidence"])
                if direction in (-1, 1) and confidence >= cfg.confidence_threshold:
                    atr_pct = float(features.loc[timestamp, "atr_pct"])
                    base_open = float(bar["open"])
                    fill = base_open * (1 + direction * friction)
                    atr_value = fill * atr_pct / 100
                    stop_distance = cfg.stop_atr_multiple * atr_value
                    if stop_distance > 0 and math.isfinite(stop_distance):
                        risk_amount = capital * cfg.risk_per_trade_pct / 100
                        quantity_by_risk = risk_amount / stop_distance
                        max_notional = (
                            capital * cfg.max_notional_pct / 100 * cfg.max_leverage
                        )
                        quantity_by_notional = max_notional / fill
                        quantity = min(quantity_by_risk, quantity_by_notional)
                        if quantity > 0:
                            stop = fill - direction * stop_distance
                            take_profit = fill + direction * (
                                cfg.take_profit_atr_multiple * atr_value
                            )
                            notional = quantity * fill
                            position = {
                                "direction": direction,
                                "signal_time": timestamp,
                                "entry_time": timestamp,
                                "entry_price": fill,
                                "quantity": quantity,
                                "notional": notional,
                                "stop_loss": stop,
                                "take_profit": take_profit,
                                "entry_fee": notional * fee_rate,
                                "holding_bars": 0,
                                "confidence": confidence,
                            }

            if position is not None:
                position["holding_bars"] += 1
                direction = position["direction"]
                stop_hit = (
                    float(bar["low"]) <= position["stop_loss"]
                    if direction == 1
                    else float(bar["high"]) >= position["stop_loss"]
                )
                target_hit = (
                    float(bar["high"]) >= position["take_profit"]
                    if direction == 1
                    else float(bar["low"]) <= position["take_profit"]
                )

                reason = None
                base_exit = None
                if stop_hit:  # conservative priority when both occur
                    reason = "stop_loss"
                    if direction == 1 and float(bar["open"]) < position["stop_loss"]:
                        base_exit = float(bar["open"])
                    elif direction == -1 and float(bar["open"]) > position["stop_loss"]:
                        base_exit = float(bar["open"])
                    else:
                        base_exit = position["stop_loss"]
                elif target_hit:
                    reason = "take_profit"
                    base_exit = position["take_profit"]
                elif position["holding_bars"] >= cfg.max_holding_bars:
                    reason = "max_holding"
                    base_exit = float(bar["close"])

                if reason:
                    exit_fill = base_exit * (1 - direction * friction)
                    exit_notional = position["quantity"] * exit_fill
                    exit_fee = exit_notional * fee_rate
                    gross_pnl = (
                        (exit_fill - position["entry_price"])
                        * position["quantity"]
                        * direction
                    )
                    funding_periods = position["holding_bars"] / 96.0
                    funding_cost = (
                        position["notional"]
                        * cfg.conservative_funding_bps_per_8h
                        / 10_000
                        * funding_periods
                    )
                    net_pnl = (
                        gross_pnl
                        - position["entry_fee"]
                        - exit_fee
                        - funding_cost
                    )
                    capital += net_pnl
                    trades.append(
                        {
                            **position,
                            "exit_time": timestamp,
                            "exit_price": exit_fill,
                            "exit_fee": exit_fee,
                            "funding_cost": funding_cost,
                            "gross_pnl": gross_pnl,
                            "net_pnl": net_pnl,
                            "exit_reason": reason,
                        }
                    )
                    position = None

            unrealized = 0.0
            if position is not None:
                unrealized = (
                    (float(bar["close"]) - position["entry_price"])
                    * position["quantity"]
                    * position["direction"]
                    - position["entry_fee"]
                )
            equity_rows.append(
                {
                    "timestamp": timestamp,
                    "equity": capital + unrealized,
                    "capital": capital,
                    "in_position": int(position is not None),
                }
            )

        if position is not None:
            timestamp = timeline[-1]
            bar = bars.loc[timestamp]
            direction = position["direction"]
            exit_fill = float(bar["close"]) * (1 - direction * friction)
            exit_notional = position["quantity"] * exit_fill
            exit_fee = exit_notional * fee_rate
            gross_pnl = (
                (exit_fill - position["entry_price"])
                * position["quantity"]
                * direction
            )
            funding_cost = (
                position["notional"]
                * cfg.conservative_funding_bps_per_8h
                / 10_000
                * (position["holding_bars"] / 96.0)
            )
            net_pnl = gross_pnl - position["entry_fee"] - exit_fee - funding_cost
            capital += net_pnl
            trades.append(
                {
                    **position,
                    "exit_time": timestamp,
                    "exit_price": exit_fill,
                    "exit_fee": exit_fee,
                    "funding_cost": funding_cost,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "exit_reason": "end_of_test",
                }
            )
            equity_rows[-1]["equity"] = capital
            equity_rows[-1]["capital"] = capital
            equity_rows[-1]["in_position"] = 0

        equity = pd.DataFrame(equity_rows).set_index("timestamp")
        return {
            "trades": trades,
            "equity": equity,
            "metrics": self.metrics(trades, equity),
            "config": cfg.__dict__,
        }

    def metrics(self, trades: list[dict], equity: pd.DataFrame) -> dict[str, Any]:
        cfg = self.config
        pnl = np.asarray([trade["net_pnl"] for trade in trades], dtype=float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        running_max = equity["equity"].cummax()
        drawdown = equity["equity"] / running_max - 1
        daily = equity["equity"].resample("1D").last().dropna()
        returns = daily.pct_change().dropna()
        sharpe = (
            float(returns.mean() / returns.std() * np.sqrt(365))
            if len(returns) > 1 and returns.std() > 0
            else None
        )
        total_days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400, 1)
        years = total_days / 365.25
        ending = float(equity["equity"].iloc[-1])
        cagr = (
            (ending / cfg.starting_capital) ** (1 / years) - 1
            if ending > 0 and years > 0
            else None
        )
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
        return {
            "starting_capital": cfg.starting_capital,
            "ending_equity": ending,
            "total_return_pct": (ending / cfg.starting_capital - 1) * 100,
            "cagr_pct": cagr * 100 if cagr is not None else None,
            "trades": int(len(trades)),
            "win_rate_pct": float(len(wins) / len(pnl) * 100) if len(pnl) else 0.0,
            "expectancy": float(pnl.mean()) if len(pnl) else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
            "max_drawdown_pct": float(drawdown.min() * 100),
            "sharpe_daily": sharpe,
            "total_fees": float(
                sum(t["entry_fee"] + t["exit_fee"] for t in trades)
            ),
            "total_funding_cost": float(sum(t["funding_cost"] for t in trades)),
            "exposure_pct": float(equity["in_position"].mean() * 100),
        }
