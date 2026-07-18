"""Generate purged OOS predictions and run the development-only v4 backtest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from core.market_store import MarketStore
from features.causal_builder import CausalFeatureBuilder
from models.purged_split import PurgedWalkForwardSplit, final_holdout_split
from models.v4_baselines import aligned_probabilities, make_model
from trading.v4_backtester import V4Backtester


def run(db: Path, instrument: str, model_name: str) -> dict:
    store = MarketStore(db)
    build = CausalFeatureBuilder(store).build(instrument, include_target=True)
    X = build["features"]
    y = build["target"]
    ends = build["target_end_time"]
    development, holdout = final_holdout_split(X.index, ends, 0.20)
    X_dev = X.iloc[development]
    y_dev = y.iloc[development]
    ends_dev = ends.iloc[development]

    splitter = PurgedWalkForwardSplit(
        n_splits=4,
        min_train_size=40_000,
        test_size=20_000,
    )
    predictions = []
    fold_metadata = []
    for fold in splitter.split(X_dev.index, ends_dev):
        model = make_model(model_name)
        model.fit(X_dev.iloc[fold.train_indices], y_dev.iloc[fold.train_indices])
        X_test = X_dev.iloc[fold.test_indices]
        probabilities = aligned_probabilities(model, X_test)
        predicted = model.predict(X_test).astype(int)
        confidence = probabilities.max(axis=1)
        frame = pd.DataFrame(
            {
                "prediction": predicted,
                "confidence": confidence,
                "prob_short": probabilities[:, 0],
                "prob_hold": probabilities[:, 1],
                "prob_long": probabilities[:, 2],
                "fold": fold.fold,
            },
            index=X_test.index,
        )
        predictions.append(frame)
        fold_metadata.append(
            {
                "fold": fold.fold,
                "train_samples": len(fold.train_indices),
                "test_samples": len(fold.test_indices),
                "train_end": fold.train_end_time.isoformat(),
                "test_start": fold.test_start_time.isoformat(),
                "label_overlap": bool(
                    (ends_dev.iloc[fold.train_indices] >= fold.test_start_time).any()
                ),
            }
        )

    oos = pd.concat(predictions).sort_index()
    bars = store.get_candles("okx", instrument, "5m", completed_only=True)
    result = V4Backtester().run(bars, X_dev, oos)
    return {
        "instrument": instrument,
        "model": model_name,
        "development_only": True,
        "holdout_evaluated": False,
        "holdout_samples_reserved": len(holdout),
        "folds": fold_metadata,
        "prediction_rows": len(oos),
        "prediction_distribution": {
            str(int(value)): int((oos["prediction"] == value).sum())
            for value in (-1, 0, 1)
        },
        "metrics": result["metrics"],
        "execution_config": result["config"],
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
    parser.add_argument("--model", choices=["logistic", "hist_gb"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = run(Path(args.db), args.instrument, args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result["metrics"], indent=2))
    print("predictions:", result["prediction_distribution"])
    print("output:", output)


if __name__ == "__main__":
    main()
