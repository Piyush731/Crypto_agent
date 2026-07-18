"""Purged walk-forward validation for causal v4 feature matrices."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.market_store import MarketStore
from features.causal_builder import CausalFeatureBuilder
from models.purged_split import PurgedWalkForwardSplit, final_holdout_split
from models.v4_baselines import MODEL_NAMES, evaluate_model, make_model


def mean_metric(folds: list[dict], name: str) -> float | None:
    values = [fold[name] for fold in folds if fold.get(name) is not None]
    return float(np.mean(values)) if values else None


def validate(
    db_path: Path,
    instrument_id: str,
    evaluate_holdout: bool,
    n_splits: int,
    min_train_size: int,
    test_size: int,
) -> dict:
    started = time.time()
    store = MarketStore(db_path)
    build = CausalFeatureBuilder(store).build(instrument_id, include_target=True)
    X = build["features"]
    y = build["target"]
    target_end = build["target_end_time"]

    development_indices, holdout_indices = final_holdout_split(
        X.index, target_end, holdout_fraction=0.20
    )
    X_dev = X.iloc[development_indices]
    y_dev = y.iloc[development_indices]
    ends_dev = target_end.iloc[development_indices]

    splitter = PurgedWalkForwardSplit(
        n_splits=n_splits,
        min_train_size=min_train_size,
        test_size=test_size,
    )
    folds = list(splitter.split(X_dev.index, ends_dev))
    if not folds:
        raise RuntimeError("No purged folds were produced")

    model_results = {}
    for model_name in MODEL_NAMES:
        fold_results = []
        print(f"\n{instrument_id} | {model_name}")
        for fold in folds:
            model = make_model(model_name)
            model.fit(
                X_dev.iloc[fold.train_indices],
                y_dev.iloc[fold.train_indices],
            )
            metrics = evaluate_model(
                model,
                X_dev.iloc[fold.test_indices],
                y_dev.iloc[fold.test_indices],
            )
            metrics.update(
                {
                    "fold": fold.fold,
                    "train_samples": int(len(fold.train_indices)),
                    "test_samples": int(len(fold.test_indices)),
                    "train_end": fold.train_end_time.isoformat(),
                    "test_start": fold.test_start_time.isoformat(),
                    "label_overlap": bool(
                        (
                            ends_dev.iloc[fold.train_indices]
                            >= fold.test_start_time
                        ).any()
                    ),
                }
            )
            fold_results.append(metrics)
            print(
                f"  fold={fold.fold} train={len(fold.train_indices)} "
                f"test={len(fold.test_indices)} "
                f"bal_acc={metrics['balanced_accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )

        model_results[model_name] = {
            "folds": fold_results,
            "mean_accuracy": mean_metric(fold_results, "accuracy"),
            "mean_balanced_accuracy": mean_metric(
                fold_results, "balanced_accuracy"
            ),
            "mean_macro_f1": mean_metric(fold_results, "macro_f1"),
            "mean_log_loss": mean_metric(fold_results, "log_loss"),
        }

    non_dummy = [name for name in MODEL_NAMES if name != "dummy"]
    selected = max(
        non_dummy,
        key=lambda name: model_results[name]["mean_macro_f1"] or float("-inf"),
    )

    holdout_result = None
    if evaluate_holdout:
        final_model = make_model(selected)
        final_model.fit(X.iloc[development_indices], y.iloc[development_indices])
        holdout_result = evaluate_model(
            final_model,
            X.iloc[holdout_indices],
            y.iloc[holdout_indices],
        )
        holdout_result.update(
            {
                "model": selected,
                "train_samples": int(len(development_indices)),
                "holdout_samples": int(len(holdout_indices)),
                "holdout_start": X.index[holdout_indices[0]].isoformat(),
                "holdout_end": X.index[holdout_indices[-1]].isoformat(),
                "label_overlap": bool(
                    (
                        target_end.iloc[development_indices]
                        >= X.index[holdout_indices[0]]
                    ).any()
                ),
            }
        )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instrument_id": instrument_id,
        "feature_metadata": {
            **build["metadata"],
            "first_decision": build["metadata"]["first_decision"].isoformat(),
            "last_decision": build["metadata"]["last_decision"].isoformat(),
        },
        "class_distribution": {
            str(label): count for label, count in sorted(Counter(y.tolist()).items())
        },
        "development_samples": int(len(development_indices)),
        "reserved_holdout_samples": int(len(holdout_indices)),
        "models": model_results,
        "selected_by_cv": selected,
        "holdout_evaluated": evaluate_holdout,
        "holdout": holdout_result,
        "elapsed_seconds": round(time.time() - started, 2),
        "warning": (
            "Do not tune after evaluating holdout. Trading performance is not "
            "established by classification metrics alone."
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Validate v4 causal ML baselines")
    parser.add_argument("--db", required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--min-train", type=int, default=40_000)
    parser.add_argument("--test-size", type=int, default=20_000)
    return parser.parse_args()


def main():
    args = parse_args()
    result = validate(
        Path(args.db),
        args.instrument,
        args.evaluate_holdout,
        args.splits,
        args.min_train,
        args.test_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str))

    print("\nSUMMARY")
    print("instrument:", result["instrument_id"])
    print("classes:", result["class_distribution"])
    for name, metrics in result["models"].items():
        print(
            name,
            "balanced_accuracy=",
            round(metrics["mean_balanced_accuracy"], 4),
            "macro_f1=",
            round(metrics["mean_macro_f1"], 4),
        )
    print("selected_by_cv:", result["selected_by_cv"])
    print("holdout_evaluated:", result["holdout_evaluated"])
    print("output:", output)


if __name__ == "__main__":
    main()
