"""Cross-venue validation of frozen T11 on Binance Vision USD-M archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.market_store import MarketStore
from portfolio_v4.time_series_momentum import TimeSeriesMomentum
from research.run_t11_walkforward import (
    DATA_START, EVALUATION_START, FINAL_HOLDOUT_START, INDEPENDENT_END,
    RECENT_PERIOD_START, WINDOWS, serialize_trade,
)
from trading.time_series_portfolio_engine_v4 import TimeSeriesPortfolioEngineV4

PROVIDER = "binance_vision"


def run(db: Path, output: Path):
    strategy = TimeSeriesMomentum()
    store = MarketStore(db)
    symbols = {symbol: f"{symbol}USDT" for symbol in strategy.candidate_symbols}
    start_ms = int(DATA_START.timestamp() * 1000)
    end_ms = int(INDEPENDENT_END.timestamp() * 1000)
    hourly = {
        symbol: store.get_candles(
            PROVIDER, instrument, "1h", start_ms=start_ms, end_ms=end_ms,
            completed_only=True,
        )
        for symbol, instrument in symbols.items()
    }
    five = {
        symbol: store.get_candles(
            PROVIDER, instrument, "5m",
            start_ms=int(EVALUATION_START.timestamp() * 1000), end_ms=end_ms,
            completed_only=True,
        )
        for symbol, instrument in symbols.items()
    }
    missing = [symbol for symbol in symbols if hourly[symbol].empty or five[symbol].empty]
    if missing:
        raise RuntimeError(f"Missing Binance cross-venue data: {missing}")
    for symbol in symbols:
        if hourly[symbol].index.max() >= RECENT_PERIOD_START:
            raise RuntimeError(f"Recent hourly period accidentally loaded: {symbol}")
        if five[symbol].index.max() >= RECENT_PERIOD_START:
            raise RuntimeError(f"Recent 5m period accidentally loaded: {symbol}")

    schedule = strategy.build_schedule(hourly)
    schedule = schedule[
        (schedule.index >= EVALUATION_START) & (schedule.index <= INDEPENDENT_END)
    ]
    full = TimeSeriesPortfolioEngineV4().run(
        five, schedule, strategy, INDEPENDENT_END
    )
    windows = []
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
        windows.append(
            {
                "name": name,
                "start": window_start.isoformat(),
                "end": window_end.isoformat(),
                "schedule_rows": len(window_schedule),
                "metrics": result["metrics"],
            }
        )

    payload = {
        "validation_id": "t12_binance_crossvenue_t11_frozen_v1",
        "strategy_key": "time_series_momentum_weekly_v1",
        "strategy_parameters_changed": False,
        "provider": PROVIDER,
        "symbols": list(symbols),
        "data_start": DATA_START.isoformat(),
        "evaluation_start": EVALUATION_START.isoformat(),
        "independent_end": INDEPENDENT_END.isoformat(),
        "recent_period_evaluated": False,
        "final_holdout_evaluated": False,
        "final_holdout_start": FINAL_HOLDOUT_START.isoformat(),
        "schedule_rows": len(schedule),
        "metrics": full["metrics"],
        "windows": windows,
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
    for key in [
        "validation_id", "provider", "strategy_parameters_changed",
        "recent_period_evaluated", "final_holdout_evaluated",
    ]:
        print(f"{key}: {result[key]}")
    print(json.dumps(result["metrics"], indent=2))
    print("windows:", json.dumps(result["windows"], indent=2))
    print("output:", args.output)


if __name__ == "__main__":
    main()
