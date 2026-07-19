"""Date-bounded V4-T11 independent-history and walk-forward evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.market_store import MarketStore
from portfolio_v4.time_series_momentum import TimeSeriesMomentum
from trading.time_series_portfolio_engine_v4 import TimeSeriesPortfolioEngineV4

DATA_START = pd.Timestamp("2021-01-01T00:00:00Z")
EVALUATION_START = pd.Timestamp("2021-04-05T00:00:00Z")
INDEPENDENT_END = pd.Timestamp("2024-07-17T23:55:00Z")
RECENT_PERIOD_START = pd.Timestamp("2024-07-18T00:00:00Z")
FINAL_HOLDOUT_START = pd.Timestamp("2026-02-22T15:20:00Z")
WINDOWS = (
    ("2021", pd.Timestamp("2021-04-05T00:00:00Z"), pd.Timestamp("2021-12-31T23:55:00Z")),
    ("2022", pd.Timestamp("2022-01-01T00:00:00Z"), pd.Timestamp("2022-12-31T23:55:00Z")),
    ("2023", pd.Timestamp("2023-01-01T00:00:00Z"), pd.Timestamp("2023-12-31T23:55:00Z")),
    ("2024H1", pd.Timestamp("2024-01-01T00:00:00Z"), INDEPENDENT_END),
)


def serialize_trade(trade):
    return {
        key: (value.isoformat() if hasattr(value, "isoformat") else value)
        for key, value in trade.items()
    }


def run(db: Path, output: Path):
    strategy = TimeSeriesMomentum()
    store = MarketStore(db)
    symbols = {
        symbol: f"{symbol}-USDT-SWAP"
        for symbol in strategy.candidate_symbols
    }
    start_ms = int(DATA_START.timestamp() * 1000)
    end_ms = int(INDEPENDENT_END.timestamp() * 1000)
    hourly = {
        symbol: store.get_candles(
            "okx", instrument, "1h", start_ms=start_ms, end_ms=end_ms,
            completed_only=True,
        )
        for symbol, instrument in symbols.items()
    }
    five = {
        symbol: store.get_candles(
            "okx", instrument, "5m",
            start_ms=int(EVALUATION_START.timestamp() * 1000), end_ms=end_ms,
            completed_only=True,
        )
        for symbol, instrument in symbols.items()
    }
    missing = [symbol for symbol in symbols if hourly[symbol].empty or five[symbol].empty]
    if missing:
        raise RuntimeError(f"Missing T11 independent-history data: {missing}")
    for symbol in symbols:
        if hourly[symbol].index.max() >= RECENT_PERIOD_START:
            raise RuntimeError(f"Recent period accidentally loaded for {symbol}")
        if five[symbol].index.max() >= RECENT_PERIOD_START:
            raise RuntimeError(f"Recent 5m period accidentally loaded for {symbol}")

    schedule = strategy.build_schedule(hourly)
    schedule = schedule[
        (schedule.index >= EVALUATION_START)
        & (schedule.index <= INDEPENDENT_END)
    ]
    engine = TimeSeriesPortfolioEngineV4()
    full = engine.run(five, schedule, strategy, INDEPENDENT_END)

    window_results = []
    for name, window_start, window_end in WINDOWS:
        window_bars = {
            symbol: frame[(frame.index >= window_start) & (frame.index <= window_end)]
            for symbol, frame in five.items()
        }
        window_schedule = schedule[
            (schedule.index >= window_start) & (schedule.index <= window_end)
        ]
        result = TimeSeriesPortfolioEngineV4().run(
            window_bars, window_schedule, strategy, window_end
        )
        window_results.append(
            {
                "name": name,
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
                "schedule_rows": len(window_schedule),
                "metrics": result["metrics"],
            }
        )

    payload = {
        "strategy_key": "time_series_momentum_weekly_v1",
        "strategy": strategy.metadata(),
        "symbols": list(symbols),
        "data_start": DATA_START.isoformat(),
        "evaluation_start": EVALUATION_START.isoformat(),
        "independent_end": INDEPENDENT_END.isoformat(),
        "recent_period_evaluated": False,
        "final_holdout_evaluated": False,
        "final_holdout_start": FINAL_HOLDOUT_START.isoformat(),
        "schedule_rows": len(schedule),
        "metrics": full["metrics"],
        "windows": window_results,
        "trades": [serialize_trade(trade) for trade in full["trades"]],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(Path(args.db), Path(args.output))
    print("strategy:", result["strategy_key"])
    print("symbols:", result["symbols"])
    print("schedule_rows:", result["schedule_rows"])
    print("recent_period_evaluated:", result["recent_period_evaluated"])
    print("final_holdout_evaluated:", result["final_holdout_evaluated"])
    print(json.dumps(result["metrics"], indent=2))
    print("windows:", json.dumps(result["windows"], indent=2))
    print("output:", args.output)


if __name__ == "__main__":
    main()
