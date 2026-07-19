"""Causal multi-leg shared-capital engine for V4-T11."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class TimeSeriesPortfolioEngineV4:
    def __init__(self, starting_capital: float = 10_000.0):
        self.starting_capital = starting_capital

    @staticmethod
    def execution_schedule(schedule, timeline, end):
        rows = []
        for signal_time, row in schedule.sort_index().iterrows():
            location = timeline.searchsorted(signal_time, side="right")
            if location >= len(timeline) or timeline[location] > end:
                continue
            item = row.copy()
            item["signal_time"] = signal_time
            item.name = timeline[location]
            rows.append(item)
        result = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not result.empty:
            result.index.name = "execution_time"
            result = result[~result.index.duplicated(keep="last")]
        return result

    def run(self, bars, schedule, strategy, development_end) -> dict[str, Any]:
        cfg = strategy.config
        common = None
        for frame in bars.values():
            index = frame.index[frame.index <= development_end]
            common = index if common is None else common.intersection(index)
        timeline = common.sort_values()
        if timeline.empty:
            raise ValueError("No common 5m timeline for V4-T11")
        executions = self.execution_schedule(schedule, timeline, development_end)
        capital = self.starting_capital
        positions = {}
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
            exit_fee = position["quantity"] * exit_price * fee_rate
            gross = (
                (exit_price - position["entry_price"])
                * position["quantity"] * direction
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
            current = {symbol: frame.loc[timestamp] for symbol, frame in bars.items()}
            if timestamp in executions.index:
                row = executions.loc[timestamp]
                desired = {symbol: 1 for symbol in tuple(row["long_symbols"])}
                desired.update({symbol: -1 for symbol in tuple(row["short_symbols"])})
                atr_pct = dict(row["atr_pct"])
                signed_score = dict(row["signed_score"])

                for symbol in list(positions):
                    if desired.get(symbol) != positions[symbol]["direction"]:
                        close_position(
                            symbol, timestamp, float(current[symbol]["open"]),
                            "weekly_rebalance",
                        )

                equity_at_open = capital + sum(
                    (float(current[symbol]["open"]) - position["entry_price"])
                    * position["quantity"] * position["direction"]
                    - position["entry_fee"] - position["funding_cost"]
                    for symbol, position in positions.items()
                )
                gross_limit = equity_at_open * cfg.max_gross_notional_pct / 100
                existing_notional = sum(position["notional"] for position in positions.values())
                for symbol, direction in desired.items():
                    if symbol in positions:
                        continue
                    base_price = float(current[symbol]["open"])
                    fill = base_price * (1 + direction * friction)
                    stop_distance = (
                        fill * float(atr_pct[symbol]) / 100 * cfg.stop_atr_multiple
                    )
                    risk_amount = equity_at_open * cfg.risk_per_leg_pct / 100
                    risk_quantity = risk_amount / stop_distance if stop_distance > 0 else 0
                    leg_limit = equity_at_open * cfg.max_notional_per_leg_pct / 100
                    available_gross = max(gross_limit - existing_notional, 0.0)
                    allowed_notional = min(leg_limit, available_gross)
                    quantity = min(risk_quantity, allowed_notional / fill)
                    if quantity <= 0:
                        continue
                    notional = quantity * fill
                    positions[symbol] = {
                        "direction": direction,
                        "signal_time": row["signal_time"],
                        "entry_time": timestamp,
                        "entry_price": fill,
                        "quantity": quantity,
                        "notional": notional,
                        "stop_loss": fill - direction * stop_distance,
                        "entry_fee": notional * fee_rate,
                        "funding_cost": 0.0,
                        "signal_score": float(signed_score[symbol]),
                    }
                    existing_notional += notional

            for symbol in list(positions):
                position = positions[symbol]
                bar = current[symbol]
                direction = position["direction"]
                hit = (
                    float(bar["low"]) <= position["stop_loss"]
                    if direction == 1
                    else float(bar["high"]) >= position["stop_loss"]
                )
                if hit:
                    base_exit = (
                        min(float(bar["open"]), position["stop_loss"])
                        if direction == 1
                        else max(float(bar["open"]), position["stop_loss"])
                    )
                    close_position(symbol, timestamp, base_exit, "atr_stop")
                else:
                    position["funding_cost"] += (
                        position["notional"] * cfg.funding_bps_per_8h / 10_000 / 96
                    )

            unrealized = sum(
                (float(current[symbol]["close"]) - position["entry_price"])
                * position["quantity"] * position["direction"]
                - position["entry_fee"] - position["funding_cost"]
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
                symbol, timeline[-1], float(bars[symbol].loc[timeline[-1], "close"]),
                "end_of_window",
            )
        equity_rows[-1].update({"equity": capital, "capital": capital, "positions": 0})
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
        drawdown = equity["equity"] / equity["equity"].cummax() - 1
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
