"""Run a deterministic registered strategy on development data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.market_store import MarketStore
from features.causal_builder import CausalFeatureBuilder
from models.purged_split import final_holdout_split
from strategies_v4.registry import get_strategy
from trading.strategy_engine import StrategyBacktestEngine


def run(db: Path, instrument: str, strategy_key: str) -> dict:
    store = MarketStore(db)
    strategy = get_strategy(strategy_key)
    labelled = CausalFeatureBuilder(store).build(instrument, include_target=True)
    development, holdout = final_holdout_split(
        labelled["features"].index,
        labelled["target_end_time"],
        0.20,
    )
    features = labelled["features"].iloc[development]
    signals = strategy.features_to_signals(features)
    if signals is None:
        raise RuntimeError("Strategy did not produce deterministic signals")
    bars = store.get_candles("okx", instrument, "5m", completed_only=True)
    result = StrategyBacktestEngine().run_signals(
        bars, features, signals, strategy, starting_capital=10_000.0
    )
    return {
        "instrument": instrument,
        "strategy": strategy.metadata(),
        "development_only": True,
        "holdout_evaluated": False,
        "development_samples": len(development),
        "reserved_holdout_samples": len(holdout),
        "signal_distribution": {
            "short": int((signals["direction"] == -1).sum()),
            "hold": int((signals["direction"] == 0).sum()),
            "long": int((signals["direction"] == 1).sum()),
        },
        "metrics": result["metrics"],
        "trades": [
            {
                key: (value.isoformat() if hasattr(value, "isoformat") else value)
                for key, value in trade.items()
            }
            for trade in result["trades"]
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(Path(args.db), args.instrument, args.strategy)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str))
    print("signals:", result["signal_distribution"])
    print(json.dumps(result["metrics"], indent=2))
    print("output:", output)


if __name__ == "__main__":
    main()
