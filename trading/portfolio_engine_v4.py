"""Shared-capital engine for market-neutral v4 portfolio schedules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from portfolio_v4.cross_sectional_momentum import CrossSectionalMomentum


class PortfolioEngineV4:
    def __init__(self, starting_capital: float = 10_000.0):
        self.starting_capital = starting_capital

    def run(
        self,
        bars: dict[str, pd.DataFrame],
        schedule: pd.DataFrame,
        strategy: CrossSectionalMomentum,
        development_end: pd.Timestamp,
    ) -> dict[str, Any]:
        cfg = strategy.config
        common_index = None
        for frame in bars.values():
            index = frame.index[frame.index <= development_end]
            common_index = index if common_index is None else common_index.intersection(index)
        timeline = common_index.sort_values()
        if timeline.empty:
            raise ValueError("No common 5m execution timeline")

        # A schedule timestamp is the instant when the completed 1h inputs
        # become available. Filling at the 5m open with that same timestamp
        # would use a price that existed before the signal was knowable. Map
        # every signal to the first common 5m bar strictly after availability.
        execution_rows = []
        for signal_time, row in schedule.sort_index().iterrows():
            execution_position = timeline.searchsorted(signal_time, side="right")
            if execution_position >= len(timeline):
                continue
            execution_time = timeline[execution_position]
            if execution_time > development_end:
                continue
            execution_row = row.copy()
            execution_row["signal_time"] = signal_time
            execution_row.name = execution_time
            execution_rows.append(execution_row)
        execution_schedule = (
            pd.DataFrame(execution_rows)
            if execution_rows
            else pd.DataFrame()
        )
        if not execution_schedule.empty:
            execution_schedule.index.name = "execution_time"
            execution_schedule = execution_schedule[
                ~execution_schedule.index.duplicated(keep="last")
            ]

        capital = self.starting_capital
        positions: dict[str, dict] = {}
        trades = []
        equity_rows = []
        fee_rate = cfg.fee_bps_per_side / 10_000
        friction = (
            cfg.slippage_bps_per_side + cfg.half_spread_bps_per_side
        ) / 10_000

        def close_position(symbol, timestamp, base_price, reason):
            nonlocal capital
            position = positions.pop(symbol)
            direction = position["direction"]
            exit_price = base_price * (1 - direction * friction)
            exit_notional = position["quantity"] * exit_price
            exit_fee = exit_notional * fee_rate
            gross = (
                (exit_price - position["entry_price"])
                * position["quantity"]
                * direction
            )
            net = gross - position["entry_fee"] - exit_fee - position["funding_cost"]
            capital += net
            trades.append(
                {
                    **position,
                    "symbol": symbol,
                    "exit_time": timestamp,
                    "exit_price": exit_price,
                    "exit_fee": exit_fee,
                    "gross_pnl": gross,
                    "net_pnl": net,
                    "exit_reason": reason,
                }
            )

        for timestamp in timeline:
            current_bars = {symbol: frame.loc[timestamp] for symbol, frame in bars.items()}

            if timestamp in execution_schedule.index:
                row = execution_schedule.loc[timestamp]
                signal_time = row["signal_time"]
                desired = {}
                if pd.notna(row.get("long_symbol")) and row.get("long_symbol"):
                    desired[str(row["long_symbol"])] = (1, float(row["long_atr_pct"]))
                if pd.notna(row.get("short_symbol")) and row.get("short_symbol"):
                    desired[str(row["short_symbol"])] = (-1, float(row["short_atr_pct"]))

                for symbol in list(positions):
                    desired_direction = desired.get(symbol, (0, None))[0]
                    if desired_direction != positions[symbol]["direction"]:
                        close_position(
                            symbol,
                            timestamp,
                            float(current_bars[symbol]["open"]),
                            "rebalance",
                        )

                equity_at_open = capital + sum(
                    (
                        float(current_bars[symbol]["open"]) - position["entry_price"]
                    )
                    * position["quantity"]
                    * position["direction"]
                    for symbol, position in positions.items()
                )

                for symbol, (direction, atr_pct) in desired.items():
                    if symbol in positions:
                        continue
                    open_price = float(current_bars[symbol]["open"])
                    fill = open_price * (1 + direction * friction)
                    stop_distance = fill * atr_pct / 100 * cfg.stop_atr_multiple
                    risk_amount = equity_at_open * cfg.risk_per_leg_pct / 100
                    quantity_by_risk = risk_amount / stop_distance if stop_distance > 0 else 0
                    max_notional = equity_at_open * cfg.max_notional_per_leg_pct / 100
                    quantity_by_notional = max_notional / fill
                    quantity = min(quantity_by_risk, quantity_by_notional)
                    if quantity <= 0:
                        continue
                    notional = quantity * fill
                    positions[symbol] = {
                        "direction": direction,
                        "signal_time": signal_time,
                        "entry_time": timestamp,
                        "entry_price": fill,
                        "quantity": quantity,
                        "notional": notional,
                        "stop_loss": fill - direction * stop_distance,
                        "entry_fee": notional * fee_rate,
                        "funding_cost": 0.0,
                    }

            for symbol in list(positions):
                position = positions[symbol]
                bar = current_bars[symbol]
                direction = position["direction"]
                stop_hit = (
                    float(bar["low"]) <= position["stop_loss"]
                    if direction == 1
                    else float(bar["high"]) >= position["stop_loss"]
                )
                if stop_hit:
                    base_exit = (
                        min(float(bar["open"]), position["stop_loss"])
                        if direction == 1
                        else max(float(bar["open"]), position["stop_loss"])
                    )
                    close_position(symbol, timestamp, base_exit, "stop_loss")
                else:
                    position["funding_cost"] += (
                        position["notional"]
                        * cfg.funding_bps_per_8h
                        / 10_000
                        / 96
                    )

            unrealized = sum(
                (
                    float(current_bars[symbol]["close"]) - position["entry_price"]
                )
                * position["quantity"]
                * position["direction"]
                - position["entry_fee"]
                - position["funding_cost"]
                for symbol, position in positions.items()
            )
            equity_rows.append(
                {
                    "timestamp": timestamp,
                    "equity": capital + unrealized,
                    "capital": capital,
                    "positions": len(positions),
                }
            )

        for symbol in list(positions):
            close_position(
                symbol,
                timeline[-1],
                float(bars[symbol].loc[timeline[-1], "close"]),
                "end_of_test",
            )
        equity_rows[-1]["equity"] = capital
        equity_rows[-1]["capital"] = capital
        equity_rows[-1]["positions"] = 0

        equity = pd.DataFrame(equity_rows).set_index("timestamp")
        return {
            "trades": trades,
            "equity": equity,
            "metrics": self.metrics(trades, equity),
            "strategy": strategy.metadata(),
        }

    def metrics(self, trades, equity):
        pnl = np.asarray([trade["net_pnl"] for trade in trades], dtype=float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]
        running_max = equity["equity"].cummax()
        drawdown = equity["equity"] / running_max - 1
        daily = equity["equity"].resample("1D").last().dropna()
        returns = daily.pct_change().dropna()
        sharpe = (
            float(returns.mean() / returns.std() * np.sqrt(365))
            if len(returns) > 1 and returns.std() > 0 else None
        )
        ending = float(equity["equity"].iloc[-1])
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
        return {
            "starting_capital": self.starting_capital,
            "ending_equity": ending,
            "total_return_pct": (ending / self.starting_capital - 1) * 100,
            "trades": len(trades),
            "win_rate_pct": len(wins) / len(pnl) * 100 if len(pnl) else 0.0,
            "expectancy": float(pnl.mean()) if len(pnl) else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
            "max_drawdown_pct": float(drawdown.min() * 100),
            "sharpe_daily": sharpe,
            "total_fees": float(sum(t["entry_fee"] + t["exit_fee"] for t in trades)),
            "total_funding_cost": float(sum(t["funding_cost"] for t in trades)),
            "average_positions": float(equity["positions"].mean()),
        }
