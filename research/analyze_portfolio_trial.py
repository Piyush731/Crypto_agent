"""Diagnostics and cost stress for a stored v4 portfolio trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def trade_metrics(frame: pd.DataFrame, cost_multiplier: float, capital: float):
    adjusted = frame["gross_pnl"] - cost_multiplier * (
        frame["entry_fee"] + frame["exit_fee"] + frame["funding_cost"]
    )
    wins = adjusted[adjusted > 0]
    losses = adjusted[adjusted <= 0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    curve = capital + adjusted.cumsum()
    peak = curve.cummax()
    drawdown = curve / peak - 1
    return {
        "cost_multiplier": cost_multiplier,
        "net_pnl": float(adjusted.sum()),
        "return_pct": float(adjusted.sum() / capital * 100),
        "trades": int(len(adjusted)),
        "win_rate_pct": float((adjusted > 0).mean() * 100),
        "expectancy": float(adjusted.mean()),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "sequence_drawdown_pct": float(drawdown.min() * 100),
    }


def grouped(frame: pd.DataFrame, column: str):
    rows = []
    for value, group in frame.groupby(column):
        rows.append(
            {
                column: str(value),
                "trades": len(group),
                "gross_pnl": float(group["gross_pnl"].sum()),
                "fees": float((group["entry_fee"] + group["exit_fee"]).sum()),
                "funding": float(group["funding_cost"].sum()),
                "net_pnl": float(group["net_pnl"].sum()),
                "win_rate_pct": float((group["net_pnl"] > 0).mean() * 100),
            }
        )
    return rows


def analyze(path: Path):
    data = json.loads(path.read_text())
    frame = pd.DataFrame(data["trades"])
    if frame.empty:
        raise RuntimeError("Trial has no trades")
    frame["entry_time"] = pd.to_datetime(frame["entry_time"], utc=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True)
    # Periods do not carry timezone metadata. Remove UTC explicitly so pandas
    # does not emit a warning while creating reporting-only month/quarter keys.
    reporting_exit_time = frame["exit_time"].dt.tz_localize(None)
    frame["month"] = reporting_exit_time.dt.to_period("M").astype(str)
    frame["quarter"] = reporting_exit_time.dt.to_period("Q").astype(str)
    frame["side"] = frame["direction"].map({1: "long", -1: "short"})

    capital = float(data["metrics"]["starting_capital"])
    start = frame["entry_time"].min()
    end = frame["exit_time"].max()
    years = (end - start).total_seconds() / (365.25 * 86400)
    ending = float(data["metrics"]["ending_equity"])
    cagr = (ending / capital) ** (1 / years) - 1 if years > 0 and ending > 0 else None

    monthly = grouped(frame, "month")
    monthly_net = [row["net_pnl"] for row in monthly]
    symbol_rows = grouped(frame, "symbol")
    direction_rows = grouped(frame, "side")
    regime_rows = grouped(frame, "entry_regime") if "entry_regime" in frame else []
    exit_reason_rows = grouped(frame, "exit_reason") if "exit_reason" in frame else []

    return {
        "source": str(path),
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        "years": years,
        "cagr_pct": cagr * 100 if cagr is not None else None,
        "gross_pnl": float(frame["gross_pnl"].sum()),
        "fees": float((frame["entry_fee"] + frame["exit_fee"]).sum()),
        "funding": float(frame["funding_cost"].sum()),
        "net_pnl": float(frame["net_pnl"].sum()),
        "profitable_months_pct": float(np.mean(np.asarray(monthly_net) > 0) * 100),
        "best_month_pnl": float(max(monthly_net)),
        "worst_month_pnl": float(min(monthly_net)),
        "cost_scenarios": [
            trade_metrics(frame, multiplier, capital)
            for multiplier in (1.0, 1.5, 2.0)
        ],
        "by_symbol": symbol_rows,
        "by_direction": direction_rows,
        "by_entry_regime": regime_rows,
        "by_exit_reason": exit_reason_rows,
        "by_quarter": grouped(frame, "quarter"),
        "by_month": monthly,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = analyze(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    print(json.dumps({key: result[key] for key in [
        "period_start", "period_end", "years", "cagr_pct",
        "gross_pnl", "fees", "funding", "net_pnl",
        "profitable_months_pct", "best_month_pnl", "worst_month_pnl",
        "cost_scenarios", "by_symbol", "by_direction", "by_entry_regime",
        "by_exit_reason", "by_quarter",
    ]}, indent=2))
    print("output:", output)


if __name__ == "__main__":
    main()
