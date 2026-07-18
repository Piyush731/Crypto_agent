"""Run one pre-registered shared-capital portfolio strategy on development data."""

import argparse
import json
from pathlib import Path

from core.market_store import MarketStore
from portfolio_v4.registry import get_portfolio_strategy
from trading.portfolio_engine_v4 import PortfolioEngineV4


def run(db: Path, output: Path, strategy_key: str):
    store = MarketStore(db)
    symbols = {
        "BTC": "BTC-USDT-SWAP",
        "ETH": "ETH-USDT-SWAP",
        "SOL": "SOL-USDT-SWAP",
        "BNB": "BNB-USDT-SWAP",
    }
    hourly = {
        symbol: store.get_candles("okx", instrument, "1h", completed_only=True)
        for symbol, instrument in symbols.items()
    }
    five_minute = {
        symbol: store.get_candles("okx", instrument, "5m", completed_only=True)
        for symbol, instrument in symbols.items()
    }

    common = None
    for frame in five_minute.values():
        common = frame.index if common is None else common.intersection(frame.index)
    common = common.sort_values()
    cutoff_position = int(len(common) * 0.80)
    development_end = common[cutoff_position - 1]

    strategy = get_portfolio_strategy(strategy_key)
    schedule = strategy.build_schedule(hourly)
    schedule = schedule[schedule.index <= development_end]
    result = PortfolioEngineV4().run(
        five_minute, schedule, strategy, development_end
    )

    serializable = {
        "development_only": True,
        "holdout_evaluated": False,
        "development_end": development_end.isoformat(),
        "reserved_holdout_start": common[cutoff_position].isoformat(),
        "strategy_key": strategy_key,
        "strategy": result["strategy"],
        "schedule_rows": len(schedule),
        "metrics": result["metrics"],
        "trades": [
            {
                key: (value.isoformat() if hasattr(value, "isoformat") else value)
                for key, value in trade.items()
            }
            for trade in result["trades"]
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(serializable, indent=2, default=str))
    return serializable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategy", required=True)
    args = parser.parse_args()

    result = run(Path(args.db), Path(args.output), args.strategy)
    print("strategy:", result["strategy_key"])
    print("schedule_rows:", result["schedule_rows"])
    print(json.dumps(result["metrics"], indent=2))
    print("output:", args.output)


if __name__ == "__main__":
    main()
