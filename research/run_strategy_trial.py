"""Run one registered strategy through purged CV and development OOS trading."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.market_store import MarketStore
from features.causal_builder import CausalFeatureBuilder
from models.purged_split import PurgedWalkForwardSplit, final_holdout_split
from models.v4_baselines import MODEL_NAMES, aligned_probabilities, evaluate_model, make_model
from strategies_v4.registry import get_strategy
from trading.strategy_engine import StrategyBacktestEngine


def mean(folds: list[dict], key: str):
    values = [row[key] for row in folds]
    return float(np.mean(values)) if values else None


def run_trial(db: Path, instrument: str, strategy_key: str) -> dict:
    started = time.time()
    store = MarketStore(db)
    strategy = get_strategy(strategy_key)

    causal = CausalFeatureBuilder(store).build(instrument, include_target=False)
    all_features = causal["features"]
    bars = store.get_candles("okx", instrument, "5m", completed_only=True)
    labels = strategy.build_labels(all_features, bars)

    common_index = all_features.index.intersection(labels.target.index)
    X = all_features.loc[common_index]
    y = labels.target.loc[common_index]
    target_end = labels.target_end_time.loc[common_index]

    development, holdout = final_holdout_split(X.index, target_end, 0.20)
    X_dev = X.iloc[development]
    y_dev = y.iloc[development]
    ends_dev = target_end.iloc[development]

    splitter = PurgedWalkForwardSplit(
        n_splits=4,
        min_train_size=40_000,
        test_size=20_000,
    )
    folds = list(splitter.split(X_dev.index, ends_dev))
    if not folds:
        raise RuntimeError("No purged folds produced")

    model_results = {}
    probability_frames = {}

    for model_name in MODEL_NAMES:
        fold_metrics = []
        oos_probabilities = []
        print(f"\n{strategy_key} | {instrument} | {model_name}")

        for fold in folds:
            model = make_model(model_name)
            model.fit(X_dev.iloc[fold.train_indices], y_dev.iloc[fold.train_indices])
            X_test = X_dev.iloc[fold.test_indices]
            y_test = y_dev.iloc[fold.test_indices]
            metrics = evaluate_model(model, X_test, y_test)
            metrics.update(
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
            fold_metrics.append(metrics)

            probabilities = aligned_probabilities(model, X_test)
            oos_probabilities.append(
                pd.DataFrame(
                    {
                        "prob_short": probabilities[:, 0],
                        "prob_hold": probabilities[:, 1],
                        "prob_long": probabilities[:, 2],
                    },
                    index=X_test.index,
                )
            )
            print(
                f"  fold={fold.fold} bal_acc={metrics['balanced_accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )

        model_results[model_name] = {
            "folds": fold_metrics,
            "mean_accuracy": mean(fold_metrics, "accuracy"),
            "mean_balanced_accuracy": mean(fold_metrics, "balanced_accuracy"),
            "mean_macro_f1": mean(fold_metrics, "macro_f1"),
            "mean_log_loss": mean(fold_metrics, "log_loss"),
        }
        probability_frames[model_name] = pd.concat(oos_probabilities).sort_index()

    selected = max(
        (name for name in MODEL_NAMES if name != "dummy"),
        key=lambda name: model_results[name]["mean_macro_f1"],
    )
    selected_probabilities = probability_frames[selected]
    engine_result = StrategyBacktestEngine().run(
        bars,
        X_dev,
        selected_probabilities,
        strategy,
        starting_capital=10_000.0,
    )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trial": strategy.metadata(),
        "instrument": instrument,
        "development_only": True,
        "holdout_evaluated": False,
        "development_samples": len(development),
        "reserved_holdout_samples": len(holdout),
        "class_distribution": {
            str(label): count for label, count in sorted(Counter(y.tolist()).items())
        },
        "label_metadata": labels.metadata,
        "models": model_results,
        "selected_by_cv_macro_f1": selected,
        "oos_probability_rows": len(selected_probabilities),
        "trading_metrics": engine_result["metrics"],
        "trades": [
            {
                key: (value.isoformat() if hasattr(value, "isoformat") else value)
                for key, value in trade.items()
            }
            for trade in engine_result["trades"]
        ],
        "elapsed_seconds": round(time.time() - started, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = run_trial(Path(args.db), args.instrument, args.strategy)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str))

    print("\nCLASSIFICATION")
    for name, metrics in result["models"].items():
        print(
            name,
            "balanced_accuracy=",
            round(metrics["mean_balanced_accuracy"], 4),
            "macro_f1=",
            round(metrics["mean_macro_f1"], 4),
        )
    print("selected:", result["selected_by_cv_macro_f1"])
    print("\nTRADING")
    print(json.dumps(result["trading_metrics"], indent=2))
    print("output:", output)


if __name__ == "__main__":
    main()
