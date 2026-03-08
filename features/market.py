"""
Crypto Futures AI Agent - Market Features
==========================================
Crypto-specific features beyond standard technical indicators.
Cross-asset correlation, multi-timeframe alignment, funding rate,
volume profile, and market microstructure features.

Feature Groups:
  Cross-Asset:     BTC/ETH correlation, relative strength, reference RSI
  Multi-Timeframe: 4h/1d indicators aligned to 1h entry timeframe
  Funding Rate:    Current rate, annualized, sentiment, mark-index spread
  Ticker Stats:    24h price change, volume ratio, high-low range
  Open Interest:   Current OI value
  Volume Profile:  POC distance, up/down volume ratio
  Trend Alignment: Cross-TF trend agreement score

Usage:
    from features.market import MarketFeatures
    mf = MarketFeatures()
    df_with_market = mf.calculate_all(primary_df, dataset)
"""

import numpy as np
import pandas as pd
from typing import List, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FEATURE_CONFIG, TIMEFRAMES
from core.logger import get_logger
from features.technical import TechnicalAnalyzer

logger = get_logger("features.market")


class MarketFeatures:
    """
    Generates crypto-specific market features.
    Requires full dataset from DataManager (multi-TF OHLCV, reference data,
    funding, ticker, OI).
    """

    def __init__(self):
        self.ta = TechnicalAnalyzer()
        self._feature_names: List[str] = []
        logger.info("MarketFeatures initialized")

    # ══════════════════════════════════════════════
    #  MAIN ENTRY POINT
    # ══════════════════════════════════════════════

    def calculate_all(
        self, primary_df: pd.DataFrame, dataset: dict
    ) -> pd.DataFrame:
        """
        Calculate all market features and append to primary DataFrame.

        Args:
            primary_df: Entry-timeframe OHLCV DataFrame (e.g. 1h).
                        May already contain technical indicator columns.
            dataset:    Full dataset dict from DataManager.get_full_dataset().
                        Keys used: symbol, ohlcv, reference_data, funding,
                        open_interest, ticker.

        Returns:
            Copy of primary_df with 35-50 market feature columns added.
        """
        if primary_df is None or primary_df.empty:
            logger.warning("Empty primary_df passed to calculate_all")
            return primary_df if primary_df is not None else pd.DataFrame()

        if not dataset:
            logger.warning("Empty dataset passed to calculate_all")
            return primary_df.copy()

        symbol = dataset.get("symbol", "unknown")
        logger.info(
            f"Calculating market features for {symbol} ({len(primary_df)} candles)"
        )

        result = primary_df.copy()
        original_cols = set(result.columns)

        # Each group wrapped in _safe_calc to isolate failures
        self._safe_calc(
            self._add_cross_asset_features, result, dataset, "Cross-asset"
        )
        self._safe_calc(
            self._add_multi_tf_features, result, dataset, "Multi-TF"
        )
        self._safe_calc(
            self._add_funding_features, result, dataset, "Funding"
        )
        self._safe_calc(
            self._add_ticker_features, result, dataset, "Ticker"
        )
        self._safe_calc(
            self._add_oi_features, result, dataset, "Open Interest"
        )
        self._safe_calc(
            self._add_volume_profile, result, dataset, "Volume Profile"
        )
        self._safe_calc(
            self._add_trend_alignment, result, dataset, "Trend Alignment"
        )

        self._feature_names = [
            c for c in result.columns if c not in original_cols
        ]
        logger.info(
            f"Generated {len(self._feature_names)} market feature columns"
        )

        return result

    # ══════════════════════════════════════════════
    #  SAFETY WRAPPER
    # ══════════════════════════════════════════════

    def _safe_calc(
        self, method, df: pd.DataFrame, dataset: dict, name: str
    ) -> None:
        """Call method, catch and log any error so other groups still run."""
        try:
            method(df, dataset)
        except Exception as e:
            logger.warning(f"Error calculating {name}: {e}")

    # ══════════════════════════════════════════════
    #  CROSS-ASSET FEATURES
    # ══════════════════════════════════════════════

    def _add_cross_asset_features(
        self, df: pd.DataFrame, dataset: dict
    ) -> None:
        """BTC/ETH correlation, relative strength, reference momentum."""
        reference_data = dataset.get("reference_data", {})
        symbol = dataset.get("symbol", "")

        if not reference_data:
            logger.debug("No reference data for cross-asset features")
            return

        for ref_symbol, ref_df in reference_data.items():
            if ref_symbol == symbol:
                continue
            if ref_df is None or ref_df.empty:
                continue

            prefix = ref_symbol.replace("USDT", "").lower()

            # Align reference 1h to primary index
            ref_cols = [c for c in ["close", "volume"] if c in ref_df.columns]
            if "close" not in ref_cols:
                logger.warning(f"{ref_symbol} reference data missing 'close'")
                continue

            ref_aligned = ref_df[ref_cols].reindex(df.index, method="ffill")

            nan_ratio = ref_aligned["close"].isna().mean()
            if nan_ratio > 0.5:
                logger.warning(
                    f"{ref_symbol} alignment: {nan_ratio:.0%} NaN, skipping"
                )
                continue

            ref_close = ref_aligned["close"]
            primary_close = df["close"]

            # ── Reference returns ──
            df[f"{prefix}_return_1"] = ref_close.pct_change(1) * 100.0
            df[f"{prefix}_return_5"] = ref_close.pct_change(5) * 100.0
            df[f"{prefix}_return_20"] = ref_close.pct_change(20) * 100.0

            # ── Reference RSI ──
            df[f"{prefix}_rsi_14"] = self.ta.calculate_rsi(ref_close, 14)

            # ── Reference volatility ──
            df[f"{prefix}_volatility_20"] = (
                ref_close.pct_change()
                .rolling(20, min_periods=20)
                .std()
                * 100.0
            )

            # ── Reference trend (price vs EMA-21 %) ──
            ref_ema = self.ta.calculate_ema(ref_close, 21)
            df[f"{prefix}_trend"] = (
                (ref_close - ref_ema) / ref_ema.replace(0, np.nan) * 100.0
            )

            # ── Rolling correlation ──
            pri_ret = primary_close.pct_change()
            ref_ret = ref_close.pct_change()
            df[f"corr_{prefix}_24"] = pri_ret.rolling(
                24, min_periods=20
            ).corr(ref_ret)
            df[f"corr_{prefix}_72"] = pri_ret.rolling(
                72, min_periods=50
            ).corr(ref_ret)

            # ── Relative strength (outperformance) ──
            df[f"rs_{prefix}_5"] = (
                primary_close.pct_change(5) * 100.0
                - ref_close.pct_change(5) * 100.0
            )
            df[f"rs_{prefix}_20"] = (
                primary_close.pct_change(20) * 100.0
                - ref_close.pct_change(20) * 100.0
            )

            logger.debug(f"Added {prefix} cross-asset features (10 cols)")

    # ══════════════════════════════════════════════
    #  MULTI-TIMEFRAME FEATURES
    # ══════════════════════════════════════════════

    def _add_multi_tf_features(
        self, df: pd.DataFrame, dataset: dict
    ) -> None:
        """Key indicators from 4h/1d aligned to entry timeframe."""
        ohlcv = dataset.get("ohlcv", {})
        entry_tf = TIMEFRAMES.get("entry", "1h")
        adx_period = FEATURE_CONFIG.get("adx_period", 14)

        higher_tfs = [
            TIMEFRAMES.get("swing", "4h"),
            TIMEFRAMES.get("macro", "1d"),
        ]

        for tf_key in higher_tfs:
            if tf_key == entry_tf:
                continue

            tf_df = ohlcv.get(tf_key)
            if tf_df is None or tf_df.empty:
                logger.debug(f"No {tf_key} data for multi-TF features")
                continue

            if len(tf_df) < 30:
                logger.warning(
                    f"{tf_key} has {len(tf_df)} candles (need ≥30), skipping"
                )
                continue

            prefix = f"tf_{tf_key}"

            try:
                close = tf_df["close"]

                # RSI
                rsi_14 = self.ta.calculate_rsi(close, 14)

                # Trend: price vs EMA-21
                ema_21 = self.ta.calculate_ema(close, 21)
                trend = (
                    (close - ema_21) / ema_21.replace(0, np.nan) * 100.0
                )

                # MACD histogram
                macd_df = self.ta.calculate_macd(close)

                # ADX
                adx_df = self.ta.calculate_adx(tf_df, adx_period)

                # Bollinger %B
                bb_df = self.ta.calculate_bollinger_bands(tf_df)

                # ATR %
                atr = self.ta.calculate_atr(tf_df)
                atr_pct = atr / close.replace(0, np.nan) * 100.0

                # Bundle features
                htf_features = pd.DataFrame(
                    {
                        f"{prefix}_rsi_14": rsi_14,
                        f"{prefix}_trend": trend,
                        f"{prefix}_macd_hist": macd_df["macd_histogram"],
                        f"{prefix}_adx": adx_df[f"adx_{adx_period}"],
                        f"{prefix}_bb_pct_b": bb_df["bb_pct_b"],
                        f"{prefix}_atr_pct": atr_pct,
                    },
                    index=tf_df.index,
                )

                # Align to primary (1h) index via forward fill
                aligned = htf_features.reindex(df.index, method="ffill")

                for col in aligned.columns:
                    df[col] = aligned[col]

                logger.debug(
                    f"Added {prefix} multi-TF features "
                    f"({len(aligned.columns)} cols)"
                )

            except Exception as e:
                logger.warning(f"Error computing {tf_key} indicators: {e}")

    # ══════════════════════════════════════════════
    #  FUNDING RATE FEATURES
    # ══════════════════════════════════════════════

    def _add_funding_features(
        self, df: pd.DataFrame, dataset: dict
    ) -> None:
        """Funding rate, annualized, sentiment, mark-index spread."""
        funding = dataset.get("funding")
        if not funding:
            logger.debug("No funding data available")
            return

        rate = funding.get("current_rate")
        rate_val = float(rate) if rate is not None else 0.0

        annualized = funding.get("annualized_rate")
        ann_val = float(annualized) if annualized is not None else 0.0

        df["funding_rate"] = rate_val
        df["funding_rate_abs"] = abs(rate_val)
        df["funding_annualized"] = ann_val

        # Contrarian sentiment:
        #   Positive rate → longs pay → overleveraged longs → bearish bias
        #   Negative rate → shorts pay → overleveraged shorts → bullish bias
        #   Neutral range: ±0.01% (0.0001)
        if rate_val > 0.0001:
            df["funding_sentiment"] = -1.0
        elif rate_val < -0.0001:
            df["funding_sentiment"] = 1.0
        else:
            df["funding_sentiment"] = 0.0

        # Mark-index spread (futures premium/discount)
        mark = funding.get("mark_price")
        index_p = funding.get("index_price")
        if mark is not None and index_p is not None:
            m, i = float(mark), float(index_p)
            if i > 0:
                df["mark_index_spread"] = (m - i) / i * 100.0

        # Historical funding trend (if history available)
        history = funding.get("history", [])
        if history and len(history) >= 3:
            recent_rates = []
            for h in history[-10:]:
                r = h.get("fundingRate", h.get("funding_rate"))
                if r is not None:
                    recent_rates.append(float(r))

            if len(recent_rates) >= 2:
                df["funding_rate_avg"] = float(np.mean(recent_rates))
                df["funding_rate_trend"] = (
                    recent_rates[-1] - recent_rates[0]
                )

        logger.debug(f"Funding rate: {rate_val:.6f}")

    # ══════════════════════════════════════════════
    #  TICKER / MICROSTRUCTURE FEATURES
    # ══════════════════════════════════════════════

    def _add_ticker_features(
        self, df: pd.DataFrame, dataset: dict
    ) -> None:
        """24h price change, high-low range, trade count, volume ratio."""
        ticker = dataset.get("ticker")
        if not ticker:
            logger.debug("No ticker data available")
            return

        # 24h price change %
        pchange = ticker.get("price_change_pct")
        if pchange is not None:
            df["ticker_price_change_24h"] = float(pchange)

        # 24h high-low range %
        high_24h = ticker.get("high_24h")
        low_24h = ticker.get("low_24h")
        if high_24h is not None and low_24h is not None:
            h, l = float(high_24h), float(low_24h)
            mid = (h + l) / 2.0
            if mid > 0:
                df["ticker_hl_range_24h"] = (h - l) / mid * 100.0

        # Trade count (in millions for scale)
        trade_count = ticker.get("trade_count")
        if trade_count is not None:
            df["ticker_trade_count"] = float(trade_count) / 1_000_000.0

        # Current bar volume vs 24h average per bar
        quote_vol_24h = ticker.get("quote_volume_24h")
        if quote_vol_24h is not None and "quote_volume" in df.columns:
            avg_vol_per_bar = float(quote_vol_24h) / 24.0
            if avg_vol_per_bar > 0:
                df["ticker_vol_vs_24h_avg"] = (
                    df["quote_volume"] / avg_vol_per_bar
                )

    # ══════════════════════════════════════════════
    #  OPEN INTEREST FEATURES
    # ══════════════════════════════════════════════

    def _add_oi_features(
        self, df: pd.DataFrame, dataset: dict
    ) -> None:
        """Open interest value (snapshot)."""
        oi = dataset.get("open_interest")
        if not oi:
            logger.debug("No OI data available")
            return

        oi_val = oi.get("open_interest")
        if oi_val is not None:
            df["oi_value"] = float(oi_val)

    # ══════════════════════════════════════════════
    #  VOLUME PROFILE FEATURES
    # ══════════════════════════════════════════════

    def _add_volume_profile(
        self, df: pd.DataFrame, dataset: dict
    ) -> None:
        """VPOC distance and up/down volume ratio."""
        if len(df) < 20:
            return

        close = df["close"]
        volume = df["volume"]
        period = 50

        # Volume-weighted price (simplified VPOC proxy)
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        vwp = (
            (tp * volume).rolling(period, min_periods=20).sum()
            / volume.rolling(period, min_periods=20).sum().replace(0, np.nan)
        )
        df["vpoc_distance"] = (
            (close - vwp) / close.replace(0, np.nan) * 100.0
        )

        # Up-volume vs down-volume ratio
        direction = np.sign(close.diff())
        up_vol = (
            (volume * (direction > 0).astype(float))
            .rolling(period, min_periods=20)
            .sum()
        )
        down_vol = (
            (volume * (direction < 0).astype(float))
            .rolling(period, min_periods=20)
            .sum()
        )
        total = (up_vol + down_vol).replace(0, np.nan)
        df["volume_up_ratio"] = up_vol / total

    # ══════════════════════════════════════════════
    #  TREND ALIGNMENT
    # ══════════════════════════════════════════════

    def _add_trend_alignment(
        self, df: pd.DataFrame, dataset: dict
    ) -> None:
        """Cross-timeframe trend agreement score (-1 to +1)."""
        signals = []

        # 1h trend (from technical indicators if pre-computed)
        if "price_vs_ema_21" in df.columns:
            signals.append(np.sign(df["price_vs_ema_21"]))
        elif "ema_slope_21" in df.columns:
            signals.append(np.sign(df["ema_slope_21"]))

        # 4h trend
        if "tf_4h_trend" in df.columns:
            signals.append(np.sign(df["tf_4h_trend"]))

        # 1d trend
        if "tf_1d_trend" in df.columns:
            signals.append(np.sign(df["tf_1d_trend"]))

        if len(signals) < 2:
            logger.debug("Not enough TF data for trend alignment")
            return

        aligned = pd.concat(signals, axis=1)

        # Mean of sign values: -1 (all bearish) to +1 (all bullish)
        df["trend_alignment"] = aligned.mean(axis=1)

        # All TFs agree flags
        df["trend_aligned_bullish"] = (
            (aligned > 0).all(axis=1).astype(int)
        )
        df["trend_aligned_bearish"] = (
            (aligned < 0).all(axis=1).astype(int)
        )

    # ══════════════════════════════════════════════
    #  UTILITY
    # ══════════════════════════════════════════════

    def get_feature_names(self) -> List[str]:
        """Return list of all generated market feature column names."""
        return self._feature_names.copy()

    def get_feature_count(self) -> int:
        """Return total number of generated market feature columns."""
        return len(self._feature_names)


# ══════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  MARKET FEATURES — TEST")
    print("=" * 70)

    from data.manager import DataManager
    from features.technical import TechnicalAnalyzer as TA

    # ── 1. Fetch full dataset ──
    dm = DataManager()
    print("\n📡 Fetching BTCUSDT full dataset (news=False for speed)...")
    dataset = dm.get_full_dataset("BTCUSDT", include_news=False)

    if not dataset or "ohlcv" not in dataset:
        print("❌ Failed to fetch dataset")
        sys.exit(1)

    ohlcv = dataset.get("ohlcv", {})
    entry_tf = TIMEFRAMES.get("entry", "1h")
    df_1h = ohlcv.get(entry_tf)

    if df_1h is None or df_1h.empty:
        print(f"❌ No {entry_tf} data. Keys: {list(ohlcv.keys())}")
        sys.exit(1)

    print(f"✅ Got {len(df_1h)} candles ({entry_tf})")
    print(f"   Timeframes available : {list(ohlcv.keys())}")
    ref_keys = list(dataset.get("reference_data", {}).keys())
    print(f"   Reference assets     : {ref_keys}")
    print(f"   Funding              : {'✅' if dataset.get('funding') else '❌'}")
    print(f"   Ticker               : {'✅' if dataset.get('ticker') else '❌'}")
    print(f"   Open Interest        : {'✅' if dataset.get('open_interest') else '❌'}")

    # ── 2. Run technical indicators first (needed for trend alignment) ──
    print("\n🔧 Running TechnicalAnalyzer on 1h data...")
    ta = TA()
    df_tech = ta.calculate_all(df_1h)
    print(f"   Technical features: {ta.get_indicator_count()}")

    # ── 3. Compute market features ──
    print("\n🌐 Computing market features...")
    mf = MarketFeatures()
    result = mf.calculate_all(df_tech, dataset)

    features = mf.get_feature_names()
    print(f"\n📊 Results")
    print(f"   Technical features : {ta.get_indicator_count()}")
    print(f"   Market features    : {mf.get_feature_count()}")
    print(f"   Total columns      : {len(result.columns)}")

    # ── 4. Group by category ──
    categories: Dict[str, List[str]] = {
        "Cross-Asset": [],
        "Multi-TF": [],
        "Funding": [],
        "Ticker": [],
        "OI": [],
        "Volume Profile": [],
        "Trend": [],
        "Other": [],
    }
    for f in features:
        if f.startswith(("btc_", "eth_", "sol_", "bnb_", "xrp_", "corr_", "rs_")):
            categories["Cross-Asset"].append(f)
        elif f.startswith("tf_"):
            categories["Multi-TF"].append(f)
        elif f.startswith("funding") or f.startswith("mark_index"):
            categories["Funding"].append(f)
        elif f.startswith("ticker"):
            categories["Ticker"].append(f)
        elif f.startswith("oi_"):
            categories["OI"].append(f)
        elif f.startswith(("vpoc", "volume_up")):
            categories["Volume Profile"].append(f)
        elif f.startswith("trend"):
            categories["Trend"].append(f)
        else:
            categories["Other"].append(f)

    print(f"\n📋 Features by category:")
    for cat, cols in categories.items():
        if cols:
            print(f"   {cat:>16s} ({len(cols):>2d}): {', '.join(cols)}")

    # ── 5. Sample values (last candle) ──
    last = result.iloc[-1]
    print(f"\n📈 Last candle ({result.index[-1]}):")
    for f in features:
        v = last[f]
        if pd.notna(v):
            if abs(v) > 1_000_000:
                print(f"   {f:>30s}: {v:>14,.0f}")
            else:
                print(f"   {f:>30s}: {v:>14.6f}")
        else:
            print(f"   {f:>30s}: {'NaN':>14s}")

    # ── 6. NaN analysis ──
    if features:
        nan_pct = result[features].isna().mean() * 100
        print(f"\n🔍 NaN Analysis ({len(features)} features, {len(result)} rows):")
        print(f"    0% NaN : {(nan_pct == 0).sum()}")
        print(f"   <10% NaN: {(nan_pct < 10).sum()}")
        print(f"   <30% NaN: {(nan_pct < 30).sum()}")
        print(f"   ≥30% NaN: {(nan_pct >= 30).sum()}")

        bad = nan_pct[nan_pct >= 30].sort_values(ascending=False)
        if len(bad):
            print(f"\n   ⚠️  High NaN (≥30%):")
            for fname, pct in bad.head(8).items():
                print(f"      {fname:>28s}: {pct:5.1f}%")

    # ── 7. Sanity checks ──
    print(f"\n🧪 Sanity checks:")
    ok = True

    # Correlation in [-1, 1]
    corr_cols = [f for f in features if f.startswith("corr_")]
    for c in corr_cols:
        v = last.get(c)
        if v is not None and pd.notna(v):
            passed = -1.0 <= v <= 1.0
            print(f"   {c:>30s} in [-1,1]  : {'✅' if passed else '❌'} ({v:.4f})")
            ok = ok and passed

    # Reference RSI in [0, 100]
    rsi_cols = [f for f in features if "_rsi_" in f]
    for c in rsi_cols:
        v = last.get(c)
        if v is not None and pd.notna(v):
            passed = 0 <= v <= 100
            print(f"   {c:>30s} in [0,100] : {'✅' if passed else '❌'} ({v:.2f})")
            ok = ok and passed

    # Trend alignment in [-1, 1]
    if "trend_alignment" in result.columns:
        v = last.get("trend_alignment")
        if v is not None and pd.notna(v):
            passed = -1.0 <= v <= 1.0
            print(f"   {'trend_alignment':>30s} in [-1,1]  : {'✅' if passed else '❌'} ({v:.4f})")
            ok = ok and passed

    # Volume up ratio in [0, 1]
    if "volume_up_ratio" in result.columns:
        v = last.get("volume_up_ratio")
        if v is not None and pd.notna(v):
            passed = 0.0 <= v <= 1.0
            print(f"   {'volume_up_ratio':>30s} in [0,1]   : {'✅' if passed else '❌'} ({v:.4f})")
            ok = ok and passed

    # Multi-TF RSI in [0, 100]
    htf_rsi = [f for f in features if f.startswith("tf_") and "rsi" in f]
    for c in htf_rsi:
        v = last.get(c)
        if v is not None and pd.notna(v):
            passed = 0 <= v <= 100
            print(f"   {c:>30s} in [0,100] : {'✅' if passed else '❌'} ({v:.2f})")
            ok = ok and passed

    # Feature count
    count = mf.get_feature_count()
    passed = count >= 20
    print(f"   {'feature count ≥ 20':>30s}          : {'✅' if passed else '❌'} ({count})")
    ok = ok and passed

    print(f"\n{'✅ All checks passed!' if ok else '⚠️  Some checks failed'}")
    print("=" * 70)