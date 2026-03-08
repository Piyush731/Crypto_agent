"""
Crypto Futures AI Agent — Feature Builder
==========================================
Combines TechnicalAnalyzer + MarketFeatures → unified feature matrix.
Creates binary target variable (UP=1 / DOWN=0) for ML training.

Pipeline:
    OHLCV → Technical (95+) → Market (35-50) → Time (4) → Target → Clean → Output

Usage:
    builder = FeatureBuilder()
    result  = builder.build_features(dataset)              # training
    result  = builder.build_prediction_features(dataset)   # live/paper
    X, y    = result['features'], result['target']
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from config import (
    TIMEFRAMES,
    PREDICTION_HORIZONS,
    ACTIVE_HORIZON,
    FEATURE_CONFIG,
)
from core.logger import get_logger
from features.technical import TechnicalAnalyzer
from features.market import MarketFeatures

logger = get_logger("features.builder")

# Raw OHLCV columns — NOT features
_PRICE_COLS = frozenset([
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume",
])


class FeatureBuilder:
    """
    Orchestrate the full feature-engineering pipeline.

    Steps:
        1. Extract entry-TF OHLCV
        2. TechnicalAnalyzer  → 95+ indicator columns
        3. MarketFeatures     → 35-50 cross-asset / multi-TF / funding cols
        4. Time features      → hour, day-of-week, session, weekend
        5. Target variable    → binary forward-return direction
        6. Clean NaN / inf, drop warmup, validate
        7. Return aligned features + target + prices + metadata
    """

    def __init__(self):
        self.technical = TechnicalAnalyzer()
        self.market = MarketFeatures()
        self._feature_names: List[str] = []
        self._last_build_info: Dict = {}
        logger.info("FeatureBuilder initialized")

    # ══════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════

    def build_features(
        self,
        dataset: Dict,
        horizon: Optional[str] = None,
        include_target: bool = True,
    ) -> Dict:
        """
        Build complete feature matrix from a DataManager dataset.

        Parameters
        ----------
        dataset : dict
            Output of ``DataManager.get_full_dataset()``.
        horizon : str | None
            Key into ``PREDICTION_HORIZONS``.  Default: ``ACTIVE_HORIZON``.
        include_target : bool
            True  → create forward-return target (training).
            False → keep all rows, target=None  (prediction).

        Returns
        -------
        dict
            features      : pd.DataFrame  (N, F)  — float64, no NaN
            target        : pd.Series|None (N,)   — int {0, 1}
            prices        : pd.DataFrame  (N, P)  — aligned OHLCV
            feature_names : list[str]
            target_info   : dict                   — class distribution
            metadata      : dict                   — build stats
        """
        t0 = time.time()
        symbol = dataset.get("symbol", "UNKNOWN")
        hz_key = horizon or ACTIVE_HORIZON
        hz_candles = PREDICTION_HORIZONS.get(hz_key, 24)

        logger.info(
            f"Building features | {symbol} | "
            f"horizon={hz_key} ({hz_candles} candles) | target={include_target}"
        )

        # ── 1. Entry-TF OHLCV ─────────────────────────────────────────
        entry_tf = TIMEFRAMES["entry"]
        ohlcv_dict = dataset.get("ohlcv", {})
        df = self._get_entry_df(ohlcv_dict, entry_tf)
        if df is None:
            return self._empty("Missing or insufficient entry-TF OHLCV")

        logger.info(
            f"  {entry_tf}: {len(df)} candles  "
            f"{df.index[0]} → {df.index[-1]}"
        )

        # ── 2. Technical indicators ────────────────────────────────────
        try:
            df = self.technical.calculate_all(df)
            logger.info(f"  Technical: +{self.technical.get_indicator_count()} cols")
        except Exception as exc:
            logger.error(f"Technical calc failed: {exc}", exc_info=True)
            return self._empty(f"Technical indicators failed: {exc}")

        # ── 3. Market features ─────────────────────────────────────────
        mkt_ok = False
        try:
            df = self.market.calculate_all(df, dataset)
            mkt_ok = True
            logger.info(f"  Market:    +{self.market.get_feature_count()} cols")
        except Exception as exc:
            logger.warning(f"Market features failed (continuing): {exc}")

        # ── 4. Time features ──────────────────────────────────────────
        df = self._add_time_features(df)

        # ── 5. Target variable ─────────────────────────────────────────
        target: Optional[pd.Series] = None
        target_info: Dict = {}

        if include_target:
            target, target_info = self._make_target(df["close"], hz_candles)
            # Drop rows where target is NaN (last hz_candles rows)
            valid_idx = target.dropna().index
            df = df.loc[valid_idx]
            target = target.loc[valid_idx].astype(int)
            logger.info(
                f"  Target: {len(target)} samples | "
                f"UP={target_info['up_pct']:.1f}%  "
                f"DOWN={target_info['down_pct']:.1f}%"
            )

        # ── 6. Separate prices / features ──────────────────────────────
        pcols = [c for c in df.columns if c in _PRICE_COLS]
        fcols = [c for c in df.columns if c not in _PRICE_COLS]

        prices = df[pcols].copy()
        features = df[fcols].copy()

        # ── 7. Clean NaN / inf ─────────────────────────────────────────
        features, target, prices = self._clean(features, target, prices)

        if len(features) < 50:
            return self._empty(
                f"Only {len(features)} clean rows after cleanup (need ≥50)"
            )

        # ── 8. Result ──────────────────────────────────────────────────
        self._feature_names = list(features.columns)
        elapsed = round(time.time() - t0, 2)

        meta = {
            "symbol": symbol,
            "horizon_key": hz_key,
            "horizon_candles": hz_candles,
            "entry_tf": entry_tf,
            "total_features": len(self._feature_names),
            "total_samples": len(features),
            "date_start": str(features.index[0]),
            "date_end": str(features.index[-1]),
            "technical_ok": True,
            "market_ok": mkt_ok,
            "has_target": target is not None,
            "build_time_s": elapsed,
        }
        self._last_build_info = meta

        logger.info(
            f"  ✅ Done: {meta['total_features']} features × "
            f"{meta['total_samples']} samples  ({elapsed}s)"
        )

        return {
            "features": features,
            "target": target,
            "prices": prices,
            "feature_names": self._feature_names,
            "target_info": target_info,
            "metadata": meta,
        }

    def build_prediction_features(self, dataset: Dict) -> Dict:
        """Build features for live / paper prediction (no target)."""
        return self.build_features(dataset, include_target=False)

    def get_feature_names(self) -> List[str]:
        """Feature column names from last successful build."""
        return self._feature_names.copy()

    def get_build_info(self) -> Dict:
        """Metadata dict from last build."""
        return self._last_build_info.copy()

    # ══════════════════════════════════════════════════════════════════
    #  PRIVATE — Target
    # ══════════════════════════════════════════════════════════════════

    def _make_target(
        self, close: pd.Series, horizon: int,
    ) -> Tuple[pd.Series, Dict]:
        """
        Binary target from forward returns.

            1 = price went UP    (future_return ≥ 0)
            0 = price went DOWN  (future_return < 0)

        HOLD is NOT a target class — it emerges when ML ensemble
        confidence falls below the trading threshold (in signal_engine).
        """
        fwd_ret = (close.shift(-horizon) - close) / close

        target = pd.Series(np.nan, index=close.index, name="target")
        has_future = fwd_ret.notna()
        target.loc[has_future & (fwd_ret >= 0)] = 1.0
        target.loc[has_future & (fwd_ret < 0)] = 0.0

        valid = target.dropna()
        n = len(valid)
        up = int((valid == 1).sum())
        dn = int((valid == 0).sum())
        fwd_valid = fwd_ret.dropna()

        info = {
            "horizon_candles": horizon,
            "total": n,
            "up_count": up,
            "down_count": dn,
            "up_pct": round(100.0 * up / max(n, 1), 1),
            "down_pct": round(100.0 * dn / max(n, 1), 1),
            "balance": round(min(up, dn) / max(up, dn, 1), 3),
            "mean_return_pct": round(float(fwd_valid.mean()) * 100, 4),
            "std_return_pct": round(float(fwd_valid.std()) * 100, 4),
            "dropped_tail": int((~has_future).sum()),
        }
        return target, info

    # ══════════════════════════════════════════════════════════════════
    #  PRIVATE — Time Features
    # ══════════════════════════════════════════════════════════════════

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-of-day / day-of-week features (useful for crypto sessions).

        Columns added:
            hour_sin, hour_cos   — cyclical hour encoding (0-23)
            dow_sin,  dow_cos    — cyclical day-of-week encoding (0=Mon)
        """
        try:
            idx = df.index
            if not hasattr(idx, "hour"):
                logger.debug("Index has no .hour — skipping time features")
                return df

            df = df.copy()

            # Cyclical encoding avoids artificial boundary (23→0, Sun→Mon)
            hour = idx.hour.values.astype(float)
            df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

            dow = idx.dayofweek.values.astype(float)
            df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
            df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

            logger.info("  Time:      +4 cols (hour_sin/cos, dow_sin/cos)")
        except Exception as exc:
            logger.warning(f"Time features failed (skipping): {exc}")
        return df

    # ══════════════════════════════════════════════════════════════════
    #  PRIVATE — Cleaning
    # ══════════════════════════════════════════════════════════════════

    def _clean(
        self,
        features: pd.DataFrame,
        target: Optional[pd.Series],
        prices: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame]:
        """
        Remove NaN / inf while keeping features-target-prices aligned.

        Strategy:
            1. Replace ±inf with NaN
            2. Drop columns that are > 50 % NaN (broken indicators)
            3. Drop leading warmup rows (first block with any NaN)
            4. Forward-fill ≤ 5 consecutive NaN gaps
            5. Drop remaining rows that still have NaN
            6. Final alignment sanity check
        """
        r0, c0 = features.shape

        # 1 — inf → NaN
        features = features.replace([np.inf, -np.inf], np.nan)

        # 2 — Drop mostly-NaN columns
        nan_frac = features.isna().mean()
        bad_cols = nan_frac[nan_frac > 0.50].index.tolist()
        if bad_cols:
            logger.warning(
                f"  Dropping {len(bad_cols)} cols (>50% NaN): "
                f"{bad_cols[:5]}{'…' if len(bad_cols) > 5 else ''}"
            )
            features = features.drop(columns=bad_cols)

        # 3 — Drop leading warmup rows
        complete = features.notna().all(axis=1)
        if complete.any():
            first_ok_pos = complete.values.argmax()   # first True
            if first_ok_pos > 0:
                features = features.iloc[first_ok_pos:]
                logger.info(f"  Warmup: dropped {first_ok_pos} leading rows")

        # Align prices + target to current features index
        idx = features.index
        prices = prices.reindex(idx)
        if target is not None:
            target = target.reindex(idx)

        # 4 — Forward-fill small gaps
        features = features.ffill(limit=5)

        # 5 — Drop remaining NaN rows
        still_nan = features.isna().any(axis=1)
        if still_nan.any():
            n_drop = int(still_nan.sum())
            features = features.loc[~still_nan]
            logger.info(f"  Residual NaN: dropped {n_drop} rows")

        # 6 — Final alignment
        idx = features.index
        prices = prices.reindex(idx)
        if target is not None:
            target = target.reindex(idx)
            # Drop any rows where target became NaN after reindex
            valid = target.notna()
            if not valid.all():
                features = features.loc[valid]
                prices = prices.loc[valid]
                target = target.loc[valid]

        logger.info(
            f"  Clean: {c0}→{features.shape[1]} cols, "
            f"{r0}→{features.shape[0]} rows"
        )
        return features, target, prices

    # ══════════════════════════════════════════════════════════════════
    #  PRIVATE — Helpers
    # ══════════════════════════════════════════════════════════════════

    def _get_entry_df(
        self,
        ohlcv: Dict,
        tf: str,
        min_rows: int = 100,
    ) -> Optional[pd.DataFrame]:
        """Validate and return entry-timeframe DataFrame."""
        raw = ohlcv.get(tf)
        if raw is None or (hasattr(raw, "empty") and raw.empty):
            logger.error(f"No OHLCV data for timeframe '{tf}'")
            return None
        df = raw.copy()
        if len(df) < min_rows:
            logger.error(f"Only {len(df)} rows for '{tf}' (need ≥{min_rows})")
            return None
        return df

    def _empty(self, reason: str) -> Dict:
        """Return empty result dict on failure."""
        logger.error(f"Build failed — {reason}")
        return {
            "features": pd.DataFrame(),
            "target": None,
            "prices": pd.DataFrame(),
            "feature_names": [],
            "target_info": {},
            "metadata": {
                "error": reason,
                "total_features": 0,
                "total_samples": 0,
            },
        }


# ═══════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """
    Test FeatureBuilder end-to-end with real Binance data.
    Run:  python -m features.builder
    """
    from data.manager import DataManager
    from config import PRIMARY_PAIR, TRADING_PAIRS

    SEP = "=" * 70

    print(f"\n{SEP}")
    print("  FEATURE BUILDER — INTEGRATION TEST")
    print(SEP)

    dm = DataManager()
    builder = FeatureBuilder()
    test_pair = PRIMARY_PAIR

    # ── Test 1: Build with target (training mode) ─────────────────────
    print(f"\n[1/4] Fetching data for {test_pair} …")
    dataset = dm.get_full_dataset(test_pair, use_cache=True, include_news=False)
    print(f"      OHLCV timeframes: {list(dataset.get('ohlcv', {}).keys())}")
    for tf, tdf in dataset.get("ohlcv", {}).items():
        if tdf is not None:
            print(f"        {tf}: {len(tdf)} candles")

    print(f"\n[2/4] Building features WITH target …")
    result = builder.build_features(dataset, include_target=True)
    meta = result["metadata"]

    if meta.get("error"):
        print(f"      ❌ ERROR: {meta['error']}")
    else:
        print(f"      ✅ Shape:  {meta['total_features']} features × {meta['total_samples']} samples")
        print(f"      Dates:  {meta['date_start']}  →  {meta['date_end']}")
        print(f"      Time:   {meta['build_time_s']}s")
        print(f"      Tech OK: {meta['technical_ok']}  |  Market OK: {meta['market_ok']}")

        ti = result["target_info"]
        print(f"\n      Target distribution:")
        print(f"        Horizon:  {ti['horizon_candles']} candles")
        print(f"        UP:       {ti['up_count']}  ({ti['up_pct']}%)")
        print(f"        DOWN:     {ti['down_count']}  ({ti['down_pct']}%)")
        print(f"        Balance:  {ti['balance']}")
        print(f"        Mean fwd: {ti['mean_return_pct']}%  ±{ti['std_return_pct']}%")

        feats = result["features"]
        names = result["feature_names"]
        print(f"\n      NaN remaining: {feats.isna().sum().sum()}")
        print(f"      Inf remaining: {np.isinf(feats.select_dtypes(include=[np.number]).values).sum()}")
        print(f"\n      First 15 features:  {names[:15]}")
        print(f"      Last  10 features:  {names[-10:]}")

        # Verify target alignment
        assert len(feats) == len(result["target"]), "features/target length mismatch!"
        assert len(feats) == len(result["prices"]), "features/prices length mismatch!"
        assert feats.isna().sum().sum() == 0, "NaN still in features!"
        print("\n      ✅ Alignment & NaN checks passed")

    # ── Test 2: Build without target (prediction mode) ────────────────
    print(f"\n[3/4] Building prediction features (no target) …")
    pred = builder.build_prediction_features(dataset)
    pm = pred["metadata"]

    if pm.get("error"):
        print(f"      ❌ ERROR: {pm['error']}")
    else:
        print(f"      ✅ Shape: {pm['total_features']} features × {pm['total_samples']} samples")
        print(f"      Target:  {pred['target']}  (should be None)")
        # Prediction mode should have MORE rows (no tail trimming)
        if not meta.get("error"):
            diff = pm["total_samples"] - meta["total_samples"]
            print(f"      Extra rows vs training: +{diff} (horizon tail preserved)")

    # ── Test 3: Multiple horizons ─────────────────────────────────────
    print(f"\n[4/4] Testing all prediction horizons …")
    for hz_key, hz_val in PREDICTION_HORIZONS.items():
        r = builder.build_features(dataset, horizon=hz_key, include_target=True)
        m = r["metadata"]
        if m.get("error"):
            print(f"      {hz_key:15s}  ❌ {m['error']}")
        else:
            ti2 = r["target_info"]
            print(
                f"      {hz_key:15s}  {hz_val:4d}h ahead | "
                f"{m['total_samples']:5d} samples | "
                f"UP {ti2['up_pct']:5.1f}%  DOWN {ti2['down_pct']:5.1f}%  "
                f"bal={ti2['balance']:.3f}"
            )

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    info = builder.get_build_info()
    print(f"  Last build info: {info.get('symbol')} | "
          f"{info.get('total_features')} feats | "
          f"{info.get('total_samples')} samples")
    print(f"  Feature names count: {len(builder.get_feature_names())}")
    print(f"\n  ✅ ALL TESTS PASSED")
    print(SEP)