"""Simple pre-registered multiclass baselines for v4 research."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

CLASSES = np.array([-1, 0, 1], dtype=int)
MODEL_NAMES = ("dummy", "logistic", "hist_gb")


def make_model(name: str):
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.3,
                        class_weight="balanced",
                        max_iter=1500,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
    if name == "hist_gb":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=150,
                        max_leaf_nodes=15,
                        min_samples_leaf=50,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
    raise KeyError(f"Unknown v4 baseline model: {name}")


def aligned_probabilities(model, X) -> np.ndarray:
    raw = model.predict_proba(X)
    observed = np.asarray(model.classes_, dtype=int)
    aligned = np.zeros((len(X), len(CLASSES)), dtype=float)
    for source_index, label in enumerate(observed):
        target_index = int(np.where(CLASSES == label)[0][0])
        aligned[:, target_index] = raw[:, source_index]
    row_sums = aligned.sum(axis=1, keepdims=True)
    return np.divide(
        aligned,
        row_sums,
        out=np.full_like(aligned, 1 / len(CLASSES)),
        where=row_sums > 0,
    )


def evaluate_model(model, X, y) -> dict[str, Any]:
    predictions = np.asarray(model.predict(X), dtype=int)
    probabilities = aligned_probabilities(model, X)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y,
        predictions,
        labels=CLASSES,
        zero_division=0,
    )
    return {
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=CLASSES)),
        "confusion_matrix": confusion_matrix(y, predictions, labels=CLASSES).tolist(),
        "per_class": {
            str(int(label)): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1_per_class[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(CLASSES)
        },
        "prediction_distribution": {
            str(int(label)): int((predictions == label).sum()) for label in CLASSES
        },
    }
