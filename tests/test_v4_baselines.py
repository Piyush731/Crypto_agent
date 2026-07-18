import numpy as np
import pandas as pd

from models.v4_baselines import CLASSES, evaluate_model, make_model


def sample_data(rows=600):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(rows, 8)))
    score = X[0] - 0.5 * X[1]
    y = pd.Series(np.where(score > 0.5, 1, np.where(score < -0.5, -1, 0)))
    return X, y


def test_all_baselines_fit_and_report_multiclass_metrics():
    X, y = sample_data()
    for name in ("dummy", "logistic", "hist_gb"):
        model = make_model(name)
        model.fit(X.iloc[:500], y.iloc[:500])
        metrics = evaluate_model(model, X.iloc[500:], y.iloc[500:])
        assert 0 <= metrics["balanced_accuracy"] <= 1
        assert 0 <= metrics["macro_f1"] <= 1
        assert len(metrics["confusion_matrix"]) == len(CLASSES)
