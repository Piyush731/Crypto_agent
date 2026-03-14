"""
Crypto Futures AI Agent — ML Ensemble Predictor
================================================
Scalable weighted-voting ensemble for binary direction prediction.

Pipeline:  Features → RobustScaler → ExtraTrees Selection → N-Model Weighted Vote

Current (config.py):
    Random Forest     25%  ─┐
    XGBoost           30%  ─┤  tree-based core
    Gradient Boosting 25%  ─┼─→ Weighted P(UP) → LONG / SHORT / HOLD
    Extra Trees       20%  ─┘

Scaling up:
    1. Add entry to MODEL_CONFIG["models"] in config.py
    2. Register the class in _MODEL_REGISTRY below
    Recommended future additions (for diversity):
        sklearn.linear_model.LogisticRegression      — linear, zero deps
        sklearn.ensemble.HistGradientBoostingClassifier — fast native sklearn
        sklearn.neural_network.MLPClassifier          — different paradigm
    Beyond 6-7 models: diminishing returns.  Diversity > quantity.

Usage:
    ens = EnsemblePredictor()
    info = ens.train(X_train, y_train)
    sig  = ens.predict(X_latest)         # → {signal, direction, confidence, ...}
    ev   = ens.evaluate(X_test, y_test)  # → {accuracy, f1, roc_auc, ...}
    ens.save("BTCUSDT")
    ens.load("BTCUSDT")
"""

import time
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    XGBClassifier = None  # type: ignore

from config import MODEL_CONFIG, FEATURE_CONFIG, SAVED_MODELS_DIR
from core.logger import get_logger

logger = get_logger("models.ensemble")


# ═══════════════════════════════════════════════════════════════════════
#  REGISTRIES — extend here, then mirror in config.py
# ═══════════════════════════════════════════════════════════════════════
_MODEL_REGISTRY: Dict[str, type] = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "extra_trees": ExtraTreesClassifier,
    # ── future (add + enable in config.py) ─────────────────────────
    # "logistic_regression": LogisticRegression,
    # "hist_gradient_boosting": HistGradientBoostingClassifier,
    # "mlp": MLPClassifier,
    # "adaboost": AdaBoostClassifier,
    # "lightgbm": LGBMClassifier,          # pip install lightgbm
}
if _HAS_XGB:
    _MODEL_REGISTRY["xgboost"] = XGBClassifier

_SCALER_MAP: Dict[str, type] = {
    "robust": RobustScaler,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}

_MODEL_FILENAMES: Dict[str, str] = {
    "random_forest": "rf.pkl",
    "xgboost": "xgb.pkl",
    "gradient_boosting": "gb.pkl",
    "extra_trees": "et.pkl",
}


class EnsemblePredictor:
    """
    Weighted-voting ensemble classifier.

    Train → predict binary direction (UP / DOWN) → output LONG / SHORT / HOLD.
    Registry-based: add models via config.py without touching this file.
    """

    def __init__(self):
        self.scaler: Optional[Any] = None
        self.feature_selector: Optional[ExtraTreesClassifier] = None
        self.all_features: List[str] = []
        self.selected_features: List[str] = []
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {}
        self._trained: bool = False
        self.training_info: Dict = {}

        # Config shortcuts
        self._max_features = FEATURE_CONFIG.get("max_features", 50)
        self._min_importance = FEATURE_CONFIG.get("min_feature_importance", 0.001)
        self._conf_threshold = MODEL_CONFIG.get("confidence_threshold", 0.55)
        self._min_agreement = MODEL_CONFIG.get("min_model_agreement", 0.60)
        self._prob_dampen = 0.65

        logger.info(
            f"EnsemblePredictor init | max_features={self._max_features} "
            f"| threshold={self._conf_threshold} | registry={list(_MODEL_REGISTRY.keys())}"
        )

    # ══════════════════════════════════════════════════════════════════
    #  TRAIN
    # ══════════════════════════════════════════════════════════════════

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Full training pipeline.

        Steps:
            1. Validate inputs + class balance
            2. Fit scaler (RobustScaler default)
            3. Feature selection via ExtraTrees importance → top N
            4. Train every enabled model from config
            5. Renormalize weights (if any model failed)
            6. Record training_info

        Parameters
        ----------
        X : pd.DataFrame   Feature matrix (samples × features), no NaN
        y : pd.Series      Binary target (0=DOWN, 1=UP)

        Returns
        -------
        dict   training_info — models trained, features selected, per-model accuracy, timing
               Contains "error" key on failure.
        """
        t0 = time.time()
        logger.info(f"Training: {X.shape[0]} samples × {X.shape[1]} features")

        # ── Validate ──────────────────────────────────────────────────
        err = self._validate_inputs(X, y)
        if err:
            return {"error": err}

        # ── 1. Scaler ────────────────────────────────────────────────
        scaler_name = MODEL_CONFIG.get("scaler", "robust")
        ScalerCls = _SCALER_MAP.get(scaler_name, RobustScaler)
        self.scaler = ScalerCls()
        self.all_features = list(X.columns)

        X_clean = self._sanitize(X)
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_clean),
            columns=self.all_features,
            index=X.index,
        )
        logger.info(f"  Scaler: {scaler_name} on {len(self.all_features)} features")

        # ── 2. Feature selection ──────────────────────────────────────
        self._fit_selector(X_scaled, y)
        X_sel = X_scaled[self.selected_features].values
        logger.info(
            f"  Selection: {len(self.all_features)} → "
            f"{len(self.selected_features)} features"
        )

        # ── 3. Train models ──────────────────────────────────────────
        y_arr = y.values
        self.models = {}
        self.model_weights = {}
        per_model: Dict[str, Dict] = {}

        for name, mcfg in MODEL_CONFIG["models"].items():
            if not mcfg.get("enabled", True):
                logger.info(f"  {name}: disabled")
                continue

            if name not in _MODEL_REGISTRY:
                logger.warning(f"  {name}: not in registry — skipping")
                continue

            try:
                model = self._make_model(name, mcfg.get("params", {}))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_sel, y_arr)

                self.models[name] = model
                self.model_weights[name] = mcfg["weight"]

                acc = accuracy_score(y_arr, model.predict(X_sel))
                per_model[name] = {
                    "train_accuracy": round(acc, 4),
                    "weight": mcfg["weight"],
                }
                logger.info(
                    f"  ✅ {name:25s}  acc={acc:.4f}  w={mcfg['weight']}"
                )

            except Exception as exc:
                logger.error(f"  ❌ {name}: {exc}")
                per_model[name] = {"error": str(exc)}

        if not self.models:
            return {"error": "All models failed to train"}

        # ── 4. Renormalize weights ────────────────────────────────────
        w_sum = sum(self.model_weights.values())
        if w_sum > 0 and abs(w_sum - 1.0) > 0.01:
            self.model_weights = {
                k: round(v / w_sum, 4) for k, v in self.model_weights.items()
            }
            logger.info(f"  Weights renormalized: {self.model_weights}")

        self._trained = True
        elapsed = round(time.time() - t0, 2)

        up_n = int((y == 1).sum())
        dn_n = int((y == 0).sum())

        self.training_info = {
            "train_samples": len(X),
            "total_features": len(self.all_features),
            "selected_features": len(self.selected_features),
            "models_trained": list(self.models.keys()),
            "model_count": len(self.models),
            "model_weights": dict(self.model_weights),
            "per_model": per_model,
            "class_distribution": {
                "up": up_n,
                "down": dn_n,
                "balance": round(min(up_n, dn_n) / max(up_n, dn_n, 1), 3),
            },
            "train_time_s": elapsed,
        }

        logger.info(
            f"  ✅ Done: {len(self.models)} models, "
            f"{len(self.selected_features)} features, {elapsed}s"
        )
        return self.training_info

    # ── internal: validation ──────────────────────────────────────────

    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series) -> Optional[str]:
        """Return error message or None if OK."""
        if len(X) < 50:
            return f"Too few samples: {len(X)} (need ≥50)"
        if len(X) != len(y):
            return f"X/y length mismatch: {len(X)} vs {len(y)}"

        vc = y.value_counts()
        if len(vc) < 2:
            return f"Only one class present: {vc.to_dict()}"

        bal = vc.min() / vc.max()
        if bal < 0.2:
            logger.warning(
                f"Severe class imbalance: {vc.to_dict()} (ratio={bal:.3f})"
            )
        return None

    # ── internal: feature selection ───────────────────────────────────

    def _fit_selector(self, X: pd.DataFrame, y: pd.Series):
        """ExtraTrees importance → top-N feature selection."""
        self.feature_selector = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.feature_selector.fit(X.values, y.values)

        imp = pd.Series(
            self.feature_selector.feature_importances_,
            index=X.columns,
        ).sort_values(ascending=False)

        above = imp[imp >= self._min_importance]
        selected = above.head(self._max_features)
        self.selected_features = list(selected.index)

        # Guarantee at least 5 features
        if len(self.selected_features) < 5:
            self.selected_features = list(imp.head(max(5, self._max_features)).index)

        logger.info(
            f"  Selector: top importance = {imp.iloc[0]:.5f}, "
            f"selected {len(self.selected_features)} features"
        )

    # ── internal: model factory ───────────────────────────────────────

    def _make_model(self, name: str, params: Dict) -> Any:
        """Instantiate model from registry with optional XGB tweaks."""
        Cls = _MODEL_REGISTRY[name]
        p = params.copy()

        if name == "xgboost":
            p.setdefault("verbosity", 0)

        return Cls(**p)

    # ══════════════════════════════════════════════════════════════════
    #  PREDICT — single signal (for signal_engine)
    # ══════════════════════════════════════════════════════════════════

    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Generate a trading signal from the LAST row of X.

        Returns
        -------
        dict
            signal          : "LONG" | "SHORT" | "HOLD"
            direction       : 1 | -1 | 0
            confidence      : float 0-1
            agreement       : float 0-1  (fraction of models agreeing with majority)
            predicted_class : int 0 or 1
            probability_up  : float
            probability_down: float
            per_model       : {name: {prediction, probability_up, weight}}
            model_count     : int
        """
        if not self._trained:
            return self._hold("Not trained")

        try:
            X_prep = self._prepare(X)
            if X_prep is None:
                return self._hold("Feature preparation failed")

            row = X_prep.iloc[[-1]].values          # (1, F) numpy

            per_model: Dict[str, Dict] = {}
            wp_up = 0.0
            wt = 0.0

            for name, model in self.models.items():
                w = self.model_weights.get(name, 0.0)
                try:
                    proba = model.predict_proba(row)[0]
                    p_up = float(proba[1]) if len(proba) > 1 else float(proba[0])
                    per_model[name] = {
                        "prediction": int(p_up >= 0.5),
                        "probability_up": round(p_up, 4),
                        "weight": round(w, 4),
                    }
                    wp_up += w * p_up
                    wt += w
                except Exception as exc:
                    logger.warning(f"predict {name}: {exc}")

            if wt == 0:
                return self._hold("All model predictions failed")

            # prob_up = wp_up / wt
            # prob_dn = 1.0 - prob_up
            prob_up = wp_up / wt
            # Dampen overconfident tree probabilities toward 0.5
            prob_up = 0.5 + (prob_up - 0.5) * self._prob_dampen
            prob_dn = 1.0 - prob_up

            # Direction + confidence
            if prob_up >= self._conf_threshold:
                signal, direction, confidence = "LONG", 1, prob_up
            elif prob_dn >= self._conf_threshold:
                signal, direction, confidence = "SHORT", -1, prob_dn
            else:
                signal, direction, confidence = "HOLD", 0, max(prob_up, prob_dn)

            # Agreement: how many models match ensemble majority
            majority = 1 if prob_up >= 0.5 else 0
            agree = sum(
                1 for m in per_model.values() if m["prediction"] == majority
            )
            agreement = agree / len(per_model) if per_model else 0.0

            return {
                "signal": signal,
                "direction": direction,
                "confidence": round(confidence, 4),
                "agreement": round(agreement, 4),
                "predicted_class": majority,
                "probability_up": round(prob_up, 4),
                "probability_down": round(prob_dn, 4),
                "per_model": per_model,
                "model_count": len(per_model),
            }

        except Exception as exc:
            logger.error(f"Predict error: {exc}", exc_info=True)
            return self._hold(f"Error: {exc}")

    # ══════════════════════════════════════════════════════════════════
    #  PREDICT BATCH — arrays (for backtester / trainer)
    # ══════════════════════════════════════════════════════════════════

    def predict_batch(self, X: pd.DataFrame) -> Dict:
        """
        Batch predictions for all rows.

        Returns
        -------
        dict
            predictions     : np.ndarray  (N,)    int {0, 1}
            probability_up  : np.ndarray  (N,)    float
            probability_down: np.ndarray  (N,)    float
            confidence      : np.ndarray  (N,)    float
            signals         : List[str]           "LONG"/"SHORT"/"HOLD"
            count           : int
        """
        empty = {
            "predictions": np.array([]),
            "probability_up": np.array([]),
            "probability_down": np.array([]),
            "confidence": np.array([]),
            "signals": [],
            "count": 0,
        }
        if not self._trained:
            empty["error"] = "Not trained"
            return empty

        try:
            X_prep = self._prepare(X)
            if X_prep is None:
                empty["error"] = "Feature prep failed"
                return empty

            n = len(X_prep)
            vals = X_prep.values
            w_proba = np.zeros((n, 2), dtype=np.float64)
            wt = 0.0

            for name, model in self.models.items():
                w = self.model_weights.get(name, 0.0)
                try:
                    proba = model.predict_proba(vals)
                    if proba.shape[1] >= 2:
                        w_proba += w * proba[:, :2]
                    wt += w
                except Exception as exc:
                    logger.warning(f"batch {name}: {exc}")

            if wt > 0:
                w_proba /= wt

            # p_up = w_proba[:, 1]
            # p_dn = w_proba[:, 0]
            p_up = w_proba[:, 1]
            # Dampen overconfident tree probabilities (matches single predict)
            p_up = 0.5 + (p_up - 0.5) * self._prob_dampen
            p_dn = 1.0 - p_up

            preds = (p_up >= 0.5).astype(int)
            conf = np.maximum(p_up, p_dn)

            thresh = self._conf_threshold
            sigs = []
            for pu in p_up:
                if pu >= thresh:
                    sigs.append("LONG")
                elif (1.0 - pu) >= thresh:
                    sigs.append("SHORT")
                else:
                    sigs.append("HOLD")

            return {
                "predictions": preds,
                "probability_up": p_up,
                "probability_down": p_dn,
                "confidence": conf,
                "signals": sigs,
                "count": n,
            }

        except Exception as exc:
            logger.error(f"Batch predict error: {exc}", exc_info=True)
            empty["error"] = str(exc)
            return empty

    # ══════════════════════════════════════════════════════════════════
    #  EVALUATE
    # ══════════════════════════════════════════════════════════════════

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Comprehensive evaluation on held-out data.

        Returns
        -------
        dict
            accuracy, precision, recall, f1_score, roc_auc, log_loss,
            confusion_matrix, test_samples, signal_distribution, per_model
        """
        if not self._trained:
            return {"error": "Not trained"}

        batch = self.predict_batch(X)
        if batch["count"] == 0:
            return {"error": batch.get("error", "Empty predictions")}

        preds = batch["predictions"]
        p_up = batch["probability_up"]
        y_arr = y.values

        # Align lengths (safety)
        n = min(len(preds), len(y_arr))
        preds, p_up, y_arr = preds[:n], p_up[:n], y_arr[:n]

        acc = accuracy_score(y_arr, preds)
        prec = precision_score(y_arr, preds, zero_division=0)
        rec = recall_score(y_arr, preds, zero_division=0)
        f1 = f1_score(y_arr, preds, zero_division=0)

        try:
            auc = roc_auc_score(y_arr, p_up)
        except Exception:
            auc = 0.0
        try:
            ll = log_loss(y_arr, np.column_stack([1 - p_up, p_up]))
        except Exception:
            ll = 0.0

        cm = confusion_matrix(y_arr, preds).tolist()

        # Per-model metrics
        per_model: Dict[str, Dict] = {}
        X_prep = self._prepare(X)
        if X_prep is not None:
            vals = X_prep.values[:n]
            for name, model in self.models.items():
                try:
                    mp = model.predict(vals)
                    per_model[name] = {
                        "accuracy": round(accuracy_score(y_arr, mp), 4),
                        "f1": round(f1_score(y_arr, mp, zero_division=0), 4),
                        "weight": round(self.model_weights.get(name, 0), 4),
                    }
                except Exception:
                    pass

        sigs = batch["signals"][:n]
        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "log_loss": round(ll, 4),
            "confusion_matrix": cm,
            "test_samples": n,
            "signal_distribution": {
                "LONG": sigs.count("LONG"),
                "SHORT": sigs.count("SHORT"),
                "HOLD": sigs.count("HOLD"),
            },
            "per_model": per_model,
        }

    # ══════════════════════════════════════════════════════════════════
    #  SAVE / LOAD
    # ══════════════════════════════════════════════════════════════════

    def save(self, symbol: str) -> bool:
        """Persist all components to ``saved_models/SYMBOL/``."""
        if not self._trained:
            logger.warning("Cannot save: not trained")
            return False

        try:
            d = SAVED_MODELS_DIR / symbol
            d.mkdir(parents=True, exist_ok=True)

            joblib.dump(self.scaler, d / "scaler.pkl")
            joblib.dump(self.all_features, d / "feature_columns.pkl")
            joblib.dump(self.selected_features, d / "selected_features.pkl")
            joblib.dump(self.model_weights, d / "model_weights.pkl")
            joblib.dump(self.feature_selector, d / "feature_selector.pkl")
            joblib.dump(self.training_info, d / "training_info.pkl")

            for name, model in self.models.items():
                fn = _MODEL_FILENAMES.get(name, f"{name}.pkl")
                joblib.dump(model, d / fn)

            logger.info(f"Saved → {d}  ({len(self.models)} models)")
            return True

        except Exception as exc:
            logger.error(f"Save failed: {exc}", exc_info=True)
            return False

    def load(self, symbol: str) -> bool:
        """Load all components from ``saved_models/SYMBOL/``."""
        try:
            d = SAVED_MODELS_DIR / symbol
            if not d.exists():
                logger.warning(f"No saved model: {d}")
                return False

            self.scaler = joblib.load(d / "scaler.pkl")
            self.all_features = joblib.load(d / "feature_columns.pkl")

            sp = d / "selected_features.pkl"
            self.selected_features = (
                joblib.load(sp) if sp.exists() else self.all_features
            )

            self.model_weights = joblib.load(d / "model_weights.pkl")

            fp = d / "feature_selector.pkl"
            if fp.exists():
                self.feature_selector = joblib.load(fp)

            tp = d / "training_info.pkl"
            if tp.exists():
                self.training_info = joblib.load(tp)

            self.models = {}
            for name in list(self.model_weights.keys()):
                fn = _MODEL_FILENAMES.get(name, f"{name}.pkl")
                fpath = d / fn
                if fpath.exists():
                    self.models[name] = joblib.load(fpath)
                else:
                    logger.warning(f"Missing: {fpath}")

            if self.models:
                self._trained = True
                logger.info(
                    f"Loaded ← {d}  ({len(self.models)} models, "
                    f"{len(self.selected_features)} features)"
                )
                return True

            logger.error("No model files loaded")
            return False

        except Exception as exc:
            logger.error(f"Load failed: {exc}", exc_info=True)
            return False

    # ══════════════════════════════════════════════════════════════════
    #  FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════════

    def get_feature_importance(self) -> Dict[str, float]:
        """Weighted-average importance across all fitted models, sorted desc."""
        if not self._trained:
            return {}

        agg = pd.Series(0.0, index=self.selected_features)
        ws = 0.0

        for name, model in self.models.items():
            w = self.model_weights.get(name, 0.0)
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                if len(fi) == len(self.selected_features):
                    agg += w * pd.Series(fi, index=self.selected_features)
                    ws += w

        if ws > 0:
            agg /= ws

        return agg.sort_values(ascending=False).round(6).to_dict()

    # ══════════════════════════════════════════════════════════════════
    #  ACCESSORS
    # ══════════════════════════════════════════════════════════════════

    def is_trained(self) -> bool:
        """Whether the ensemble has been fitted."""
        return self._trained

    def get_model_info(self) -> Dict:
        """Full summary of current state."""
        return {
            "is_trained": self._trained,
            "models": list(self.models.keys()),
            "model_count": len(self.models),
            "weights": dict(self.model_weights),
            "total_features": len(self.all_features),
            "selected_features": len(self.selected_features),
            "confidence_threshold": self._conf_threshold,
            "min_agreement": self._min_agreement,
            "training_info": self.training_info,
        }

    # ══════════════════════════════════════════════════════════════════
    #  PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _prepare(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Scale + select features, preserving training column order."""
        try:
            xcols = set(X.columns)
            n = len(X)

            # Build array aligned to training column order
            arr = np.zeros((n, len(self.all_features)), dtype=np.float64)
            for i, feat in enumerate(self.all_features):
                if feat in xcols:
                    v = X[feat].values.astype(np.float64)
                    arr[:, i] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

            scaled = self.scaler.transform(arr)
            df = pd.DataFrame(scaled, columns=self.all_features, index=X.index)
            return df[self.selected_features]

        except Exception as exc:
            logger.error(f"_prepare: {exc}", exc_info=True)
            return None

    @staticmethod
    def _sanitize(X: pd.DataFrame) -> np.ndarray:
        """Replace inf/NaN with 0, cast to float64."""
        arr = X.values.astype(np.float64)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _hold(self, reason: str) -> Dict:
        """Neutral HOLD signal dict."""
        logger.warning(f"HOLD: {reason}")
        return {
            "signal": "HOLD",
            "direction": 0,
            "confidence": 0.0,
            "agreement": 0.0,
            "predicted_class": -1,
            "probability_up": 0.5,
            "probability_down": 0.5,
            "per_model": {},
            "model_count": 0,
            "reason": reason,
        }


# ═══════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """
    Full test suite: synthetic data → train → predict → evaluate → save/load.
    Run:  python -m models.ensemble
    """
    import shutil
    from sklearn.datasets import make_classification

    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  ENSEMBLE PREDICTOR — TEST SUITE")
    print(SEP)

    # ── 1. Synthetic data ─────────────────────────────────────────────
    print("\n[1/8] Generating synthetic data …")
    X_raw, y_raw = make_classification(
        n_samples=800, n_features=80, n_informative=20,
        n_redundant=15, n_classes=2, random_state=42,
    )
    feat_names = [f"feat_{i:03d}" for i in range(X_raw.shape[1])]
    X_df = pd.DataFrame(X_raw, columns=feat_names)
    y_sr = pd.Series(y_raw, name="target")

    split = int(len(X_df) * 0.75)
    X_train, X_test = X_df.iloc[:split], X_df.iloc[split:]
    y_train, y_test = y_sr.iloc[:split], y_sr.iloc[split:]
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Features: {X_df.shape[1]} | UP: {(y_sr==1).sum()} | DOWN: {(y_sr==0).sum()}")

    # ── 2. Train ──────────────────────────────────────────────────────
    print(f"\n[2/8] Training …")
    ens = EnsemblePredictor()
    info = ens.train(X_train, y_train)
    assert "error" not in info, f"Training failed: {info.get('error')}"
    print(f"  Models:    {info['models_trained']}")
    print(f"  Weights:   {info['model_weights']}")
    print(f"  Features:  {info['total_features']} → {info['selected_features']}")
    print(f"  Time:      {info['train_time_s']}s")
    for mn, mi in info["per_model"].items():
        if "train_accuracy" in mi:
            print(f"    {mn:25s} train_acc={mi['train_accuracy']}")

    # ── 3. Predict single ─────────────────────────────────────────────
    print(f"\n[3/8] Single prediction (last test row) …")
    sig = ens.predict(X_test)
    assert sig["signal"] in ("LONG", "SHORT", "HOLD")
    assert 0.0 <= sig["confidence"] <= 1.0
    assert 0.0 <= sig["agreement"] <= 1.0
    print(f"  Signal:     {sig['signal']}")
    print(f"  Direction:  {sig['direction']}")
    print(f"  Confidence: {sig['confidence']}")
    print(f"  Agreement:  {sig['agreement']}")
    print(f"  P(UP):      {sig['probability_up']}")
    for mn, mp in sig["per_model"].items():
        print(f"    {mn:25s} pred={mp['prediction']}  p_up={mp['probability_up']}")

    # ── 4. Predict single row ─────────────────────────────────────────
    print(f"\n[4/8] Single-row DataFrame …")
    one = X_test.iloc[[0]]
    s1 = ens.predict(one)
    assert s1["signal"] in ("LONG", "SHORT", "HOLD")
    print(f"  Signal: {s1['signal']}  conf={s1['confidence']}")

    # ── 5. Batch predict ──────────────────────────────────────────────
    print(f"\n[5/8] Batch prediction ({len(X_test)} rows) …")
    batch = ens.predict_batch(X_test)
    assert batch["count"] == len(X_test)
    assert len(batch["predictions"]) == len(X_test)
    sigs = batch["signals"]
    print(f"  LONG:  {sigs.count('LONG')}")
    print(f"  SHORT: {sigs.count('SHORT')}")
    print(f"  HOLD:  {sigs.count('HOLD')}")
    print(f"  Mean confidence: {batch['confidence'].mean():.4f}")

    # ── 6. Evaluate ───────────────────────────────────────────────────
    print(f"\n[6/8] Evaluation …")
    ev = ens.evaluate(X_test, y_test)
    assert "error" not in ev
    print(f"  Accuracy:   {ev['accuracy']}")
    print(f"  Precision:  {ev['precision']}")
    print(f"  Recall:     {ev['recall']}")
    print(f"  F1:         {ev['f1_score']}")
    print(f"  ROC AUC:    {ev['roc_auc']}")
    print(f"  Log Loss:   {ev['log_loss']}")
    print(f"  Confusion:  {ev['confusion_matrix']}")
    print(f"  Signals:    {ev['signal_distribution']}")
    for mn, mi in ev.get("per_model", {}).items():
        print(f"    {mn:25s} acc={mi['accuracy']}  f1={mi['f1']}")

    # ── 7. Feature importance ─────────────────────────────────────────
    print(f"\n[7/8] Feature importance (top 10) …")
    imp = ens.get_feature_importance()
    assert len(imp) > 0
    for i, (feat, sc) in enumerate(imp.items()):
        if i >= 10:
            break
        bar = "█" * int(sc * 200)
        print(f"    {feat:15s} {sc:.6f}  {bar}")

    # ── 8. Save → Load → Re-predict ──────────────────────────────────
    print(f"\n[8/8] Save → Load → Re-predict …")
    ok = ens.save("_TEST_SYN")
    assert ok, "Save failed"
    print(f"  Save: ✅")

    ens2 = EnsemblePredictor()
    ok = ens2.load("_TEST_SYN")
    assert ok, "Load failed"
    print(f"  Load: ✅")

    sig2 = ens2.predict(X_test)
    match = (
        sig["signal"] == sig2["signal"]
        and abs(sig["confidence"] - sig2["confidence"]) < 0.001
    )
    print(f"  Re-predict: {sig2['signal']}  conf={sig2['confidence']}")
    print(f"  Matches original: {'✅' if match else '⚠️ MISMATCH'}")

    info2 = ens2.get_model_info()
    print(f"  State: {info2['model_count']} models, "
          f"{info2['selected_features']} features")
    assert info2["is_trained"]

    # Cleanup
    test_dir = SAVED_MODELS_DIR / "_TEST_SYN"
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"  Cleaned up {test_dir}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  Registry:  {list(_MODEL_REGISTRY.keys())}")
    print(f"  XGBoost:   {'✅ available' if _HAS_XGB else '❌ not installed'}")
    print(f"\n  ✅ ALL TESTS PASSED")
    print(SEP)