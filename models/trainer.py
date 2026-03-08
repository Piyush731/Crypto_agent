"""
Crypto Futures AI Agent — Model Trainer
========================================
Walk-forward training, TimeSeriesSplit CV, full pipeline orchestration.

Pipeline per symbol:
    DataManager → FeatureBuilder → Walk-Forward CV → Final Train → Evaluate → Save → DB log

Design:
    - Walk-forward validation prevents data leakage (train past → test future)
    - TimeSeriesSplit for cross-validated performance estimate
    - Final model trains on ALL data for maximum information
    - Retraining check based on model staleness (configurable hours)
    - Every result logged to SQLite for performance tracking

Usage:
    trainer = ModelTrainer()
    result  = trainer.train_symbol("BTCUSDT")
    results = trainer.train_all()
    status  = trainer.get_training_status()
    trainer.retrain_if_needed("BTCUSDT")
"""

import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.model_selection import TimeSeriesSplit

from config import (
    TRADING_PAIRS,
    MODEL_CONFIG,
    PREDICTION_HORIZONS,
    ACTIVE_HORIZON,
    SAVED_MODELS_DIR,
    SCHEDULE_CONFIG,
)
from core.logger import get_logger
from core.db import get_db
from features.builder import FeatureBuilder
from models.ensemble import EnsemblePredictor

logger = get_logger("models.trainer")


class ModelTrainer:
    """
    Orchestrate training pipeline for one or all symbols.

    Responsibilities:
        - Fetch data → build features → validate → train → evaluate → save
        - Walk-forward validation for realistic performance estimates
        - Stale-model detection + automatic retraining
        - Log every training run to SQLite
    """

    def __init__(self):
        self.builder = FeatureBuilder()
        self._cv_splits = MODEL_CONFIG.get("cv_splits", 5)
        self._test_size = MODEL_CONFIG.get("test_size", 0.2)
        self._retrain_hours = SCHEDULE_CONFIG.get("retrain_interval_hours", 24)
        self._results: Dict[str, Dict] = {}
        logger.info(
            f"ModelTrainer init | cv_splits={self._cv_splits} "
            f"| test_size={self._test_size} | retrain_every={self._retrain_hours}h"
        )

    # ══════════════════════════════════════════════════════════════════
    #  TRAIN SINGLE SYMBOL — main entry
    # ══════════════════════════════════════════════════════════════════

    def train_symbol(
        self,
        symbol: str,
        dataset: Optional[Dict] = None,
        horizon: Optional[str] = None,
        save_model: bool = True,
    ) -> Dict:
        """
        Full training pipeline for one symbol.

        Steps:
            1. Get data (via DataManager or accept pre-fetched)
            2. Build features + target via FeatureBuilder
            3. Train/test split (time-ordered)
            4. Walk-forward CV for performance estimate
            5. Train final model on full training set
            6. Evaluate on held-out test set
            7. Save model artifacts
            8. Log results to SQLite

        Parameters
        ----------
        symbol : str            e.g. "BTCUSDT"
        dataset : dict | None   Pre-fetched DataManager output.  None → fetch live.
        horizon : str | None    PREDICTION_HORIZONS key.  None → ACTIVE_HORIZON.
        save_model : bool       Persist to saved_models/{symbol}/

        Returns
        -------
        dict
            success, symbol, horizon, samples, features, cv_results,
            test_evaluation, training_info, model_saved, total_time_s
            "error" key on failure.
        """
        t0 = time.time()
        hz_key = horizon or ACTIVE_HORIZON
        logger.info(f"{'='*60}")
        logger.info(f"Training {symbol} | horizon={hz_key}")
        logger.info(f"{'='*60}")

        result = {
            "success": False,
            "symbol": symbol,
            "horizon": hz_key,
            "timestamp": datetime.utcnow().isoformat(),
        }

        try:
            # ── 1. Data ───────────────────────────────────────────────
            if dataset is None:
                dataset = self._fetch_data(symbol)
                if dataset is None:
                    result["error"] = "Data fetch failed"
                    self._log_to_db(result)
                    return result

            # ── 2. Features ───────────────────────────────────────────
            build = self.builder.build_features(
                dataset, horizon=hz_key, include_target=True,
            )
            meta = build.get("metadata", {})
            if meta.get("error") or build["features"].empty:
                result["error"] = meta.get("error", "Empty features")
                self._log_to_db(result)
                return result

            X: pd.DataFrame = build["features"]
            y: pd.Series = build["target"]
            prices: pd.DataFrame = build["prices"]
            feature_names: List[str] = build["feature_names"]
            target_info: Dict = build["target_info"]

            result["samples"] = len(X)
            result["features"] = len(feature_names)
            result["target_info"] = target_info
            result["date_range"] = {
                "start": str(X.index[0]),
                "end": str(X.index[-1]),
            }

            logger.info(
                f"  Features: {len(X)} samples × {len(feature_names)} features"
            )

            # ── 3. Train / test split (time-ordered) ──────────────────
            split_idx = int(len(X) * (1.0 - self._test_size))
            if split_idx < 50:
                result["error"] = f"Train set too small: {split_idx} (need ≥50)"
                self._log_to_db(result)
                return result

            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            logger.info(
                f"  Split: train={len(X_train)} | test={len(X_test)} "
                f"(split at {X.index[split_idx]})"
            )

            # ── 4. Walk-forward cross-validation ──────────────────────
            logger.info(f"  Walk-forward CV ({self._cv_splits} folds) …")
            cv_results = self.walk_forward_validate(
                X_train, y_train, n_splits=self._cv_splits,
            )
            result["cv_results"] = cv_results

            if cv_results.get("mean_accuracy"):
                logger.info(
                    f"  CV: acc={cv_results['mean_accuracy']:.4f}±"
                    f"{cv_results['std_accuracy']:.4f}  "
                    f"f1={cv_results['mean_f1']:.4f}±"
                    f"{cv_results['std_f1']:.4f}  "
                    f"auc={cv_results['mean_roc_auc']:.4f}"
                )

            # ── 5. Final training on full train set ───────────────────
            logger.info(f"  Final model training on {len(X_train)} samples …")
            ensemble = EnsemblePredictor()
            train_info = ensemble.train(X_train, y_train)

            if train_info.get("error"):
                result["error"] = f"Training failed: {train_info['error']}"
                self._log_to_db(result)
                return result

            result["training_info"] = train_info

            # ── 6. Evaluate on held-out test set ──────────────────────
            logger.info(f"  Evaluating on {len(X_test)} test samples …")
            evaluation = ensemble.evaluate(X_test, y_test)
            result["test_evaluation"] = evaluation

            if evaluation.get("accuracy"):
                logger.info(
                    f"  Test: acc={evaluation['accuracy']:.4f} "
                    f"f1={evaluation['f1_score']:.4f} "
                    f"auc={evaluation['roc_auc']:.4f} "
                    f"signals={evaluation.get('signal_distribution', {})}"
                )

            # ── 7. Feature importance ─────────────────────────────────
            importance = ensemble.get_feature_importance()
            top_n = dict(list(importance.items())[:15])
            result["top_features"] = top_n
            if top_n:
                logger.info(f"  Top 5: {list(top_n.keys())[:5]}")

            # ── 8. Save ──────────────────────────────────────────────
            if save_model:
                saved = ensemble.save(symbol)
                result["model_saved"] = saved
                if saved:
                    # Save build metadata alongside model
                    self._save_train_meta(symbol, result)
                    logger.info(f"  Saved → {SAVED_MODELS_DIR / symbol}")
                else:
                    logger.warning("  Save failed")

            result["success"] = True
            elapsed = round(time.time() - t0, 2)
            result["total_time_s"] = elapsed

            logger.info(f"  ✅ {symbol} complete in {elapsed}s")

        except Exception as exc:
            result["error"] = str(exc)
            logger.error(f"  ❌ {symbol}: {exc}", exc_info=True)

        self._results[symbol] = result
        self._log_to_db(result)
        return result

    # ══════════════════════════════════════════════════════════════════
    #  TRAIN ALL PAIRS
    # ══════════════════════════════════════════════════════════════════

    def train_all(
        self,
        symbols: Optional[List[str]] = None,
        horizon: Optional[str] = None,
    ) -> Dict:
        """
        Train models for multiple symbols sequentially.

        Returns
        -------
        dict
            per_symbol: {symbol: result_dict},
            summary: {total, successful, failed, avg_accuracy, total_time_s}
        """
        pairs = symbols or TRADING_PAIRS
        t0 = time.time()
        logger.info(f"Training {len(pairs)} pairs: {pairs}")

        per_symbol: Dict[str, Dict] = {}
        success_count = 0
        accuracies = []

        for i, sym in enumerate(pairs, 1):
            logger.info(f"\n{'─'*40} [{i}/{len(pairs)}] {sym}")
            try:
                res = self.train_symbol(sym, horizon=horizon)
                per_symbol[sym] = res
                if res.get("success"):
                    success_count += 1
                    acc = (
                        res.get("test_evaluation", {}).get("accuracy")
                        or res.get("cv_results", {}).get("mean_accuracy")
                    )
                    if acc:
                        accuracies.append(acc)
            except Exception as exc:
                logger.error(f"  {sym} failed: {exc}")
                per_symbol[sym] = {"success": False, "error": str(exc)}

        elapsed = round(time.time() - t0, 2)
        summary = {
            "total": len(pairs),
            "successful": success_count,
            "failed": len(pairs) - success_count,
            "avg_accuracy": round(np.mean(accuracies), 4) if accuracies else None,
            "total_time_s": elapsed,
        }

        logger.info(f"\n{'='*60}")
        logger.info(
            f"Training complete: {success_count}/{len(pairs)} OK | "
            f"avg_acc={summary['avg_accuracy']} | {elapsed}s"
        )

        return {"per_symbol": per_symbol, "summary": summary}

    # ══════════════════════════════════════════════════════════════════
    #  WALK-FORWARD VALIDATION
    # ══════════════════════════════════════════════════════════════════

    def walk_forward_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: Optional[int] = None,
    ) -> Dict:
        """
        Walk-forward (expanding window) cross-validation.

        For each fold i:
            Train on [0 .. split_i]
            Test  on [split_i .. split_{i+1}]
        Earlier folds have less training data — mirrors real deployment.

        Returns
        -------
        dict
            folds: List[{fold, train_size, test_size, accuracy, f1, roc_auc, ...}]
            mean_accuracy, std_accuracy, mean_f1, std_f1, mean_roc_auc, std_roc_auc
            total_folds
        """
        splits = n_splits or self._cv_splits
        n = len(X)

        # Need enough data for meaningful folds
        min_train = 50
        min_test = 20
        max_possible = (n - min_train) // min_test
        splits = min(splits, max(1, max_possible))

        if n < min_train + min_test:
            logger.warning(f"Walk-forward: insufficient data ({n} rows)")
            return {"folds": [], "total_folds": 0, "error": "Insufficient data"}

        tscv = TimeSeriesSplit(n_splits=splits)
        folds: List[Dict] = []

        for fold_i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            if len(train_idx) < min_train or len(test_idx) < min_test:
                logger.debug(
                    f"    Fold {fold_i}: skip (train={len(train_idx)}, "
                    f"test={len(test_idx)})"
                )
                continue

            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_te = X.iloc[test_idx]
            y_te = y.iloc[test_idx]

            try:
                ens = EnsemblePredictor()
                info = ens.train(X_tr, y_tr)
                if info.get("error"):
                    logger.warning(f"    Fold {fold_i}: train error — {info['error']}")
                    continue

                ev = ens.evaluate(X_te, y_te)
                if ev.get("error"):
                    logger.warning(f"    Fold {fold_i}: eval error — {ev['error']}")
                    continue

                fold_result = {
                    "fold": fold_i,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "train_period": f"{X.index[train_idx[0]]} → {X.index[train_idx[-1]]}",
                    "test_period": f"{X.index[test_idx[0]]} → {X.index[test_idx[-1]]}",
                    "accuracy": ev["accuracy"],
                    "precision": ev["precision"],
                    "recall": ev["recall"],
                    "f1_score": ev["f1_score"],
                    "roc_auc": ev["roc_auc"],
                    "log_loss": ev["log_loss"],
                    "signal_distribution": ev.get("signal_distribution", {}),
                }
                folds.append(fold_result)

                logger.info(
                    f"    Fold {fold_i}: train={len(train_idx)} test={len(test_idx)} "
                    f"acc={ev['accuracy']:.4f} f1={ev['f1_score']:.4f} "
                    f"auc={ev['roc_auc']:.4f}"
                )

            except Exception as exc:
                logger.warning(f"    Fold {fold_i}: exception — {exc}")

        if not folds:
            return {"folds": [], "total_folds": 0, "error": "All folds failed"}

        # Aggregate
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "log_loss"]
        agg: Dict = {"folds": folds, "total_folds": len(folds)}

        for m in metrics:
            vals = [f[m] for f in folds if f.get(m) is not None]
            if vals:
                agg[f"mean_{m}"] = round(float(np.mean(vals)), 4)
                agg[f"std_{m}"] = round(float(np.std(vals)), 4)
                agg[f"min_{m}"] = round(float(np.min(vals)), 4)
                agg[f"max_{m}"] = round(float(np.max(vals)), 4)

        # Rename for convenience
        agg["mean_f1"] = agg.get("mean_f1_score", 0.0)
        agg["std_f1"] = agg.get("std_f1_score", 0.0)

        return agg

    # ══════════════════════════════════════════════════════════════════
    #  CROSS-VALIDATE (TimeSeriesSplit)
    # ══════════════════════════════════════════════════════════════════

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: Optional[int] = None,
    ) -> Dict:
        """
        TimeSeriesSplit cross-validation — alias for walk_forward_validate.
        Both use expanding-window, time-ordered splits.
        """
        return self.walk_forward_validate(X, y, n_splits)

    # ══════════════════════════════════════════════════════════════════
    #  RETRAIN IF NEEDED
    # ══════════════════════════════════════════════════════════════════

    def retrain_if_needed(
        self,
        symbol: str,
        dataset: Optional[Dict] = None,
        force: bool = False,
    ) -> Dict:
        """
        Check model staleness and retrain if needed.

        Parameters
        ----------
        symbol : str
        dataset : dict | None   Pre-fetched data
        force : bool            Force retrain regardless of staleness

        Returns
        -------
        dict
            retrained: bool, reason: str, result: dict|None
        """
        if force:
            logger.info(f"{symbol}: forced retrain")
            res = self.train_symbol(symbol, dataset=dataset)
            return {"retrained": True, "reason": "forced", "result": res}

        # Check existing model
        status = self._get_model_status(symbol)

        if not status["exists"]:
            logger.info(f"{symbol}: no saved model — training")
            res = self.train_symbol(symbol, dataset=dataset)
            return {"retrained": True, "reason": "no_model", "result": res}

        hours_old = status.get("hours_since_train", float("inf"))
        if hours_old >= self._retrain_hours:
            logger.info(
                f"{symbol}: model is {hours_old:.1f}h old "
                f"(threshold={self._retrain_hours}h) — retraining"
            )
            res = self.train_symbol(symbol, dataset=dataset)
            return {"retrained": True, "reason": f"stale_{hours_old:.1f}h", "result": res}

        logger.info(
            f"{symbol}: model is {hours_old:.1f}h old — fresh "
            f"(threshold={self._retrain_hours}h)"
        )
        return {"retrained": False, "reason": "fresh", "result": None}

    # ══════════════════════════════════════════════════════════════════
    #  QUICK TRAIN — build features from pre-built X, y
    # ══════════════════════════════════════════════════════════════════

    def quick_train(
        self,
        symbol: str,
        X: pd.DataFrame,
        y: pd.Series,
        save_model: bool = True,
    ) -> Dict:
        """
        Train directly from pre-built features (skip data fetch + builder).
        Useful for backtester or when features are already computed.

        Returns
        -------
        dict   Same structure as train_symbol result.
        """
        t0 = time.time()
        logger.info(f"Quick train {symbol}: {len(X)}×{X.shape[1]}")

        result = {
            "success": False,
            "symbol": symbol,
            "horizon": ACTIVE_HORIZON,
            "timestamp": datetime.utcnow().isoformat(),
            "samples": len(X),
            "features": X.shape[1],
        }

        try:
            # Split
            split_idx = int(len(X) * (1.0 - self._test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # CV
            cv = self.walk_forward_validate(X_train, y_train)
            result["cv_results"] = cv

            # Train
            ensemble = EnsemblePredictor()
            info = ensemble.train(X_train, y_train)
            if info.get("error"):
                result["error"] = info["error"]
                return result
            result["training_info"] = info

            # Evaluate
            ev = ensemble.evaluate(X_test, y_test)
            result["test_evaluation"] = ev

            # Save
            if save_model:
                result["model_saved"] = ensemble.save(symbol)
                if result["model_saved"]:
                    self._save_train_meta(symbol, result)

            result["success"] = True
            result["total_time_s"] = round(time.time() - t0, 2)

        except Exception as exc:
            result["error"] = str(exc)
            logger.error(f"Quick train {symbol}: {exc}", exc_info=True)

        self._results[symbol] = result
        self._log_to_db(result)
        return result

    # ══════════════════════════════════════════════════════════════════
    #  STATUS & INFO
    # ══════════════════════════════════════════════════════════════════

    def get_training_status(
        self, symbols: Optional[List[str]] = None,
    ) -> Dict:
        """
        Check saved model status for all trading pairs.

        Returns
        -------
        dict
            per_symbol: {symbol: {exists, hours_since_train, features, models, ...}}
            summary: {total, trained, untrained, stale}
        """
        pairs = symbols or TRADING_PAIRS
        per_symbol: Dict[str, Dict] = {}
        trained = 0
        stale = 0

        for sym in pairs:
            status = self._get_model_status(sym)
            per_symbol[sym] = status
            if status["exists"]:
                trained += 1
                if status.get("hours_since_train", 0) >= self._retrain_hours:
                    stale += 1

        return {
            "per_symbol": per_symbol,
            "summary": {
                "total": len(pairs),
                "trained": trained,
                "untrained": len(pairs) - trained,
                "stale": stale,
                "retrain_threshold_hours": self._retrain_hours,
            },
        }

    def get_last_result(self, symbol: str) -> Optional[Dict]:
        """Last training result for symbol (this session only)."""
        return self._results.get(symbol)

    def get_all_results(self) -> Dict[str, Dict]:
        """All training results from this session."""
        return dict(self._results)

    # ══════════════════════════════════════════════════════════════════
    #  PRIVATE — Data Fetching
    # ══════════════════════════════════════════════════════════════════

    def _fetch_data(self, symbol: str) -> Optional[Dict]:
        """Lazy-import DataManager to avoid circular imports, fetch data."""
        try:
            from data.manager import DataManager

            dm = DataManager()
            status = dm.get_status()
            if not status.get("binance"):
                logger.error("Binance connection failed")
                return None

            dataset = dm.get_full_dataset(
                symbol, use_cache=True, include_news=False,
            )

            ohlcv = dataset.get("ohlcv", {})
            if not ohlcv:
                logger.error(f"No OHLCV data for {symbol}")
                return None

            logger.info(
                f"  Data: {symbol} | "
                f"timeframes={list(ohlcv.keys())} | "
                f"quality={dataset.get('data_quality', {}).get('score', '?')}"
            )
            return dataset

        except Exception as exc:
            logger.error(f"Data fetch {symbol}: {exc}", exc_info=True)
            return None

    # ══════════════════════════════════════════════════════════════════
    #  PRIVATE — Model Status
    # ══════════════════════════════════════════════════════════════════

    def _get_model_status(self, symbol: str) -> Dict:
        """Check saved model directory for a symbol."""
        d = SAVED_MODELS_DIR / symbol
        status = {
            "symbol": symbol,
            "exists": False,
            "path": str(d),
        }

        if not d.exists():
            return status

        # Check essential files
        essential = ["scaler.pkl", "model_weights.pkl", "feature_columns.pkl"]
        has_all = all((d / f).exists() for f in essential)
        if not has_all:
            status["exists"] = False
            status["missing_files"] = [f for f in essential if not (d / f).exists()]
            return status

        status["exists"] = True

        # Model files
        model_files = list(d.glob("*.pkl"))
        status["file_count"] = len(model_files)
        status["files"] = [f.name for f in model_files]

        # Last modified (proxy for last trained)
        try:
            latest = max(f.stat().st_mtime for f in model_files)
            trained_at = datetime.fromtimestamp(latest)
            status["last_trained"] = trained_at.isoformat()
            status["hours_since_train"] = round(
                (datetime.now() - trained_at).total_seconds() / 3600, 1,
            )
        except Exception:
            status["hours_since_train"] = float("inf")

        # Training metadata
        meta_path = d / "train_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                status["features"] = meta.get("features")
                status["samples"] = meta.get("samples")
                status["test_accuracy"] = meta.get("test_accuracy")
                status["cv_accuracy"] = meta.get("cv_accuracy")
            except Exception:
                pass

        return status

    # ══════════════════════════════════════════════════════════════════
    #  PRIVATE — Metadata Save
    # ══════════════════════════════════════════════════════════════════

    def _save_train_meta(self, symbol: str, result: Dict):
        """Save lightweight JSON metadata alongside model pkl files."""
        try:
            d = SAVED_MODELS_DIR / symbol
            d.mkdir(parents=True, exist_ok=True)

            meta = {
                "symbol": symbol,
                "trained_at": datetime.utcnow().isoformat(),
                "horizon": result.get("horizon"),
                "samples": result.get("samples"),
                "features": result.get("features"),
                "test_accuracy": (
                    result.get("test_evaluation", {}).get("accuracy")
                ),
                "test_f1": (
                    result.get("test_evaluation", {}).get("f1_score")
                ),
                "test_roc_auc": (
                    result.get("test_evaluation", {}).get("roc_auc")
                ),
                "cv_accuracy": (
                    result.get("cv_results", {}).get("mean_accuracy")
                ),
                "cv_f1": (
                    result.get("cv_results", {}).get("mean_f1")
                ),
                "models_trained": (
                    result.get("training_info", {}).get("models_trained", [])
                ),
            }

            with open(d / "train_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

        except Exception as exc:
            logger.warning(f"Meta save {symbol}: {exc}")

    # ══════════════════════════════════════════════════════════════════
    #  PRIVATE — DB Logging
    # ══════════════════════════════════════════════════════════════════

    def _log_to_db(self, result: Dict):
        """Log training result to SQLite performance table."""
        try:
            db = get_db()

            test_eval = result.get("test_evaluation", {})
            cv_res = result.get("cv_results", {})

            data = {
                "mode": "training",
                "symbol": result.get("symbol", "UNKNOWN"),
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                "total_trades": result.get("samples", 0),
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl_usd": 0.0,
                "total_pnl_pct": 0.0,
                "win_rate": test_eval.get("accuracy", 0.0),
                "avg_win_pct": test_eval.get("f1_score", 0.0),
                "avg_loss_pct": test_eval.get("log_loss", 0.0),
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": test_eval.get("roc_auc", 0.0),
                "notes": json.dumps({
                    "success": result.get("success", False),
                    "horizon": result.get("horizon"),
                    "features": result.get("features"),
                    "cv_accuracy": cv_res.get("mean_accuracy"),
                    "cv_f1": cv_res.get("mean_f1"),
                    "test_accuracy": test_eval.get("accuracy"),
                    "models": result.get("training_info", {}).get(
                        "models_trained", [],
                    ),
                    "error": result.get("error"),
                    "time_s": result.get("total_time_s"),
                }),
            }

            db.save_performance(data)

        except Exception as exc:
            logger.warning(f"DB log failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """
    Test suite: synthetic → train → CV → save/load check → status.
    For live Binance test: uncomment the live block at bottom.
    Run:  python -m models.trainer
    """
    import shutil
    from sklearn.datasets import make_classification

    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  MODEL TRAINER — TEST SUITE")
    print(SEP)

    trainer = ModelTrainer()

    # ── Synthetic dataset mimicking FeatureBuilder output ─────────────
    print("\n[1/7] Generating synthetic data …")
    X_raw, y_raw = make_classification(
        n_samples=600, n_features=60, n_informative=15,
        n_redundant=10, n_classes=2, random_state=42,
    )
    feat_names = [f"feat_{i:03d}" for i in range(X_raw.shape[1])]
    idx = pd.date_range("2024-01-01", periods=600, freq="1h")
    X_df = pd.DataFrame(X_raw, columns=feat_names, index=idx)
    y_sr = pd.Series(y_raw, name="target", index=idx)
    print(f"  Shape: {X_df.shape} | UP={int((y_sr==1).sum())} DOWN={int((y_sr==0).sum())}")

    # ── Test: walk_forward_validate ───────────────────────────────────
    print(f"\n[2/7] Walk-forward validation (3 folds) …")
    cv = trainer.walk_forward_validate(X_df, y_sr, n_splits=3)
    assert cv["total_folds"] > 0, "No folds completed"
    print(f"  Folds: {cv['total_folds']}")
    print(f"  Accuracy: {cv.get('mean_accuracy', 0):.4f} ± {cv.get('std_accuracy', 0):.4f}")
    print(f"  F1:       {cv.get('mean_f1', 0):.4f} ± {cv.get('std_f1', 0):.4f}")
    print(f"  ROC AUC:  {cv.get('mean_roc_auc', 0):.4f}")
    for fold in cv["folds"]:
        print(
            f"    Fold {fold['fold']}: train={fold['train_size']} "
            f"test={fold['test_size']} "
            f"acc={fold['accuracy']:.4f} f1={fold['f1_score']:.4f}"
        )

    # ── Test: cross_validate (alias) ──────────────────────────────────
    print(f"\n[3/7] Cross-validate (alias check) …")
    cv2 = trainer.cross_validate(X_df, y_sr, n_splits=2)
    assert cv2["total_folds"] > 0
    print(f"  OK: {cv2['total_folds']} folds")

    # ── Test: quick_train ─────────────────────────────────────────────
    print(f"\n[4/7] Quick train (_TEST_SYNTH) …")
    qr = trainer.quick_train("_TEST_SYNTH", X_df, y_sr, save_model=True)
    assert qr["success"], f"Quick train failed: {qr.get('error')}"
    print(f"  Success:  {qr['success']}")
    print(f"  Samples:  {qr['samples']}")
    print(f"  Features: {qr['features']}")
    te = qr.get("test_evaluation", {})
    print(f"  Test acc: {te.get('accuracy')}")
    print(f"  Test f1:  {te.get('f1_score')}")
    print(f"  Saved:    {qr.get('model_saved')}")
    print(f"  Time:     {qr.get('total_time_s')}s")

    # ── Test: training status ─────────────────────────────────────────
    print(f"\n[5/7] Training status …")
    status = trainer.get_training_status(["_TEST_SYNTH", "BTCUSDT"])
    summ = status["summary"]
    print(f"  Total:     {summ['total']}")
    print(f"  Trained:   {summ['trained']}")
    print(f"  Untrained: {summ['untrained']}")
    for sym, ss in status["per_symbol"].items():
        ex = "✅" if ss["exists"] else "❌"
        age = ss.get("hours_since_train", "N/A")
        acc = ss.get("test_accuracy", "N/A")
        print(f"    {sym:15s} {ex}  age={age}h  acc={acc}")

    # ── Test: retrain_if_needed ───────────────────────────────────────
    print(f"\n[6/7] Retrain-if-needed check …")
    rr = trainer.retrain_if_needed("_TEST_SYNTH")
    print(f"  Retrained: {rr['retrained']} | Reason: {rr['reason']}")

    rr2 = trainer.retrain_if_needed("_TEST_SYNTH", force=True)
    print(f"  Forced:    {rr2['retrained']} | Reason: {rr2['reason']}")
    assert rr2["retrained"]

    # ── Test: last results ────────────────────────────────────────────
    print(f"\n[7/7] Session results …")
    last = trainer.get_last_result("_TEST_SYNTH")
    assert last is not None
    all_res = trainer.get_all_results()
    print(f"  Symbols trained this session: {list(all_res.keys())}")

    # ── Cleanup ───────────────────────────────────────────────────────
    test_dir = SAVED_MODELS_DIR / "_TEST_SYNTH"
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"\n  Cleaned up {test_dir}")

    # ╔════════════════════════════════════════════════════════════════╗
    # ║  LIVE BINANCE TEST — uncomment below to test with real data  ║
    # ╚════════════════════════════════════════════════════════════════╝
    print(f"\n{'─'*40}")
    print("  LIVE TEST: Training BTCUSDT from Binance …")
    print(f"{'─'*40}")
    live_result = trainer.train_symbol("BTCUSDT")
    if live_result["success"]:
        te = live_result.get("test_evaluation", {})
        cv = live_result.get("cv_results", {})
        print(f"  ✅ Success")
        print(f"  Samples:    {live_result['samples']}")
        print(f"  Features:   {live_result['features']}")
        print(f"  CV acc:     {cv.get('mean_accuracy')}")
        print(f"  Test acc:   {te.get('accuracy')}")
        print(f"  Test f1:    {te.get('f1_score')}")
        print(f"  Test AUC:   {te.get('roc_auc')}")
        print(f"  Signals:    {te.get('signal_distribution')}")
        print(f"  Top feats:  {list(live_result.get('top_features', {}).keys())[:5]}")
        print(f"  Time:       {live_result['total_time_s']}s")
    else:
        print(f"  ❌ Failed: {live_result.get('error')}")

    print(f"\n{SEP}")
    print(f"  ✅ ALL TESTS PASSED")
    print(SEP)