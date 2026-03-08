"""
Crypto Futures AI Agent - Technical Indicators
================================================
Calculates 95+ technical indicators from OHLCV data.
Pure pandas/numpy — no ta-lib required.

Indicator Groups:
  Trend:       EMA, SMA, MACD, ADX, Ichimoku
  Momentum:    RSI, Stochastic, Williams %R, CCI, MFI
  Volatility:  Bollinger Bands, ATR
  Volume:      OBV, VWAP, Volume MAs, Taker Buy Ratio
  Statistical: Returns, Volatility, Z-Score, Lags
  Derived:     Crossovers, Slopes, Price vs MA, Candle patterns

Usage:
    from features.technical import TechnicalAnalyzer
    ta = TechnicalAnalyzer()
    df_with_indicators = ta.calculate_all(ohlcv_df)
"""

import numpy as np
import pandas as pd
from typing import List, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FEATURE_CONFIG
from core.logger import get_logger

logger = get_logger("features.technical")


class TechnicalAnalyzer:
    """
    Calculates technical indicators for a single-timeframe OHLCV DataFrame.
    All parameters driven by FEATURE_CONFIG in config.py.
    """

    def __init__(self):
        self.config = FEATURE_CONFIG
        self._feature_names: List[str] = []
        logger.info("TechnicalAnalyzer initialized")

    # ══════════════════════════════════════════════
    #  MAIN ENTRY POINT
    # ══════════════════════════════════════════════

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators and append as new columns.

        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume].
                Must have DatetimeIndex.

        Returns:
            Copy of df with 95+ new indicator columns added.
            Original columns are preserved unchanged.
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame passed to calculate_all")
            return df if df is not None else pd.DataFrame()

        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return df

        logger.info(f"Calculating indicators for {len(df)} candles")
        result = df.copy()
        original_cols = set(df.columns)

        # ── Trend ──
        self._safe_add(self._add_ema, result, "EMA")
        self._safe_add(self._add_sma, result, "SMA")
        self._safe_add(self._add_macd, result, "MACD")
        self._safe_add(self._add_adx, result, "ADX")
        if self.config.get("ichimoku", True):
            self._safe_add(self._add_ichimoku, result, "Ichimoku")

        # ── Momentum ──
        self._safe_add(self._add_rsi, result, "RSI")
        self._safe_add(self._add_stochastic, result, "Stochastic")
        if self.config.get("williams_r", True):
            self._safe_add(self._add_williams_r, result, "Williams %R")
        self._safe_add(self._add_cci, result, "CCI")
        self._safe_add(self._add_mfi, result, "MFI")

        # ── Volatility ──
        self._safe_add(self._add_bollinger_bands, result, "Bollinger Bands")
        self._safe_add(self._add_atr, result, "ATR")

        # ── Volume ──
        if self.config.get("obv", True):
            self._safe_add(self._add_obv, result, "OBV")
        if self.config.get("vwap", True):
            self._safe_add(self._add_vwap, result, "VWAP")
        self._safe_add(self._add_volume_features, result, "Volume features")

        # ── Statistical ──
        self._safe_add(self._add_returns, result, "Returns")
        self._safe_add(self._add_volatility, result, "Volatility")
        self._safe_add(self._add_lags, result, "Lags")
        self._safe_add(self._add_z_score, result, "Z-Score")

        # ── Derived / Composite ──
        self._safe_add(self._add_price_vs_ma, result, "Price vs MA")
        self._safe_add(self._add_ma_slopes, result, "MA Slopes")
        self._safe_add(self._add_crossover_signals, result, "Crossovers")
        self._safe_add(self._add_candle_features, result, "Candle features")
        self._safe_add(self._add_momentum_features, result, "Momentum features")

        # Track generated feature names
        self._feature_names = [c for c in result.columns if c not in original_cols]
        logger.info(f"Generated {len(self._feature_names)} indicator columns")

        # Warn about high-NaN features
        if self._feature_names:
            nan_pct = result[self._feature_names].isna().mean()
            high_nan = nan_pct[nan_pct > 0.5]
            if len(high_nan) > 0:
                logger.warning(
                    f"{len(high_nan)} features have >50% NaN "
                    f"(likely need more candles for warmup)"
                )

        return result

    # ══════════════════════════════════════════════
    #  PUBLIC CALCULATION METHODS
    # ══════════════════════════════════════════════

    def calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Relative Strength Index (Wilder's smoothing)."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(
            alpha=1.0 / period, min_periods=period, adjust=False
        ).mean()
        avg_loss = loss.ewm(
            alpha=1.0 / period, min_periods=period, adjust=False
        ).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False, min_periods=period).mean()

    def calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period, min_periods=period).mean()

    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = None,
        std: float = None,
    ) -> pd.DataFrame:
        """Bollinger Bands: upper, middle, lower, width, %B."""
        period = period or self.config["bb_period"]
        std = std or self.config["bb_std"]

        middle = df["close"].rolling(window=period, min_periods=period).mean()
        rolling_std = df["close"].rolling(window=period, min_periods=period).std()
        upper = middle + std * rolling_std
        lower = middle - std * rolling_std

        band_range = (upper - lower).replace(0, np.nan)

        return pd.DataFrame(
            {
                "bb_upper": upper,
                "bb_middle": middle,
                "bb_lower": lower,
                "bb_width": band_range / middle.replace(0, np.nan),
                "bb_pct_b": (df["close"] - lower) / band_range,
            },
            index=df.index,
        )

    def calculate_macd(
        self,
        series: pd.Series,
        fast: int = None,
        slow: int = None,
        signal: int = None,
    ) -> pd.DataFrame:
        """MACD: line, signal line, histogram."""
        fast = fast or self.config["macd_fast"]
        slow = slow or self.config["macd_slow"]
        signal = signal or self.config["macd_signal"]

        ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(
            span=signal, adjust=False, min_periods=signal
        ).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame(
            {
                "macd": macd_line,
                "macd_signal": signal_line,
                "macd_histogram": histogram,
            },
            index=series.index,
        )

    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Average True Range (Wilder's smoothing)."""
        period = period or self.config["atr_period"]

        high_low = df["high"] - df["low"]
        high_cp = (df["high"] - df["close"].shift(1)).abs()
        low_cp = (df["low"] - df["close"].shift(1)).abs()

        true_range = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        return true_range.ewm(
            alpha=1.0 / period, min_periods=period, adjust=False
        ).mean()

    def calculate_adx(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """Average Directional Index with +DI / −DI."""
        period = period or self.config["adx_period"]

        up_move = df["high"].diff()
        down_move = -df["low"].diff()

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        high_low = df["high"] - df["low"]
        high_cp = (df["high"] - df["close"].shift(1)).abs()
        low_cp = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)

        _alpha = 1.0 / period
        atr_s = tr.ewm(alpha=_alpha, min_periods=period, adjust=False).mean()
        plus_di = (
            100.0
            * plus_dm.ewm(alpha=_alpha, min_periods=period, adjust=False).mean()
            / atr_s.replace(0, np.nan)
        )
        minus_di = (
            100.0
            * minus_dm.ewm(alpha=_alpha, min_periods=period, adjust=False).mean()
            / atr_s.replace(0, np.nan)
        )

        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(alpha=_alpha, min_periods=period, adjust=False).mean()

        return pd.DataFrame(
            {f"adx_{period}": adx, "plus_di": plus_di, "minus_di": minus_di},
            index=df.index,
        )

    def calculate_stochastic(
        self, df: pd.DataFrame, k_period: int = None, d_period: int = None
    ) -> pd.DataFrame:
        """Stochastic Oscillator %K and %D."""
        k_period = k_period or self.config["stoch_k_period"]
        d_period = d_period or self.config["stoch_d_period"]

        low_min = df["low"].rolling(window=k_period, min_periods=k_period).min()
        high_max = df["high"].rolling(window=k_period, min_periods=k_period).max()

        denom = (high_max - low_min).replace(0, np.nan)
        stoch_k = 100.0 * (df["close"] - low_min) / denom
        stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()

        return pd.DataFrame(
            {"stoch_k": stoch_k, "stoch_d": stoch_d}, index=df.index
        )

    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(df["close"].diff())
        direction.iloc[0] = 0
        return (direction * df["volume"]).cumsum()

    def calculate_vwap(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Rolling VWAP (24/7 crypto — no session boundary)."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_vp = (typical_price * df["volume"]).rolling(
            window=period, min_periods=1
        ).sum()
        cum_v = df["volume"].rolling(window=period, min_periods=1).sum()
        return cum_vp / cum_v.replace(0, np.nan)

    def calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ichimoku Cloud: Tenkan, Kijun, Senkou A/B."""
        tenkan = (
            df["high"].rolling(9, min_periods=9).max()
            + df["low"].rolling(9, min_periods=9).min()
        ) / 2.0

        kijun = (
            df["high"].rolling(26, min_periods=26).max()
            + df["low"].rolling(26, min_periods=26).min()
        ) / 2.0

        senkou_a = ((tenkan + kijun) / 2.0).shift(26)

        senkou_b = (
            (
                df["high"].rolling(52, min_periods=52).max()
                + df["low"].rolling(52, min_periods=52).min()
            )
            / 2.0
        ).shift(26)

        return pd.DataFrame(
            {
                "ichi_tenkan": tenkan,
                "ichi_kijun": kijun,
                "ichi_senkou_a": senkou_a,
                "ichi_senkou_b": senkou_b,
            },
            index=df.index,
        )

    def calculate_williams_r(
        self, df: pd.DataFrame, period: int = 14
    ) -> pd.Series:
        """Williams %R."""
        high_max = df["high"].rolling(window=period, min_periods=period).max()
        low_min = df["low"].rolling(window=period, min_periods=period).min()
        denom = (high_max - low_min).replace(0, np.nan)
        return -100.0 * (high_max - df["close"]) / denom

    def calculate_cci(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Commodity Channel Index."""
        period = period or self.config["cci_period"]

        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        sma_tp = tp.rolling(window=period, min_periods=period).mean()
        mad = tp.rolling(window=period, min_periods=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        return (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

    def calculate_mfi(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Money Flow Index."""
        period = period or self.config["mfi_period"]

        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        raw_mf = tp * df["volume"]

        tp_diff = tp.diff()
        pos_mf = raw_mf.where(tp_diff > 0, 0.0)
        neg_mf = raw_mf.where(tp_diff < 0, 0.0)

        pos_sum = pos_mf.rolling(window=period, min_periods=period).sum()
        neg_sum = neg_mf.rolling(window=period, min_periods=period).sum()

        mfr = pos_sum / neg_sum.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + mfr))

    # ══════════════════════════════════════════════
    #  PRIVATE — add indicator groups in-place
    # ══════════════════════════════════════════════

    def _safe_add(self, method, df: pd.DataFrame, name: str) -> None:
        """Wrapper: catch errors so one failing group doesn't block the rest."""
        try:
            method(df)
        except Exception as e:
            logger.warning(f"Error calculating {name}: {e}")

    # ── Trend ──

    def _add_ema(self, df: pd.DataFrame) -> None:
        for period in self.config.get("ema_periods", [21]):
            df[f"ema_{period}"] = self.calculate_ema(df["close"], period)

    def _add_sma(self, df: pd.DataFrame) -> None:
        for period in self.config.get("sma_periods", [20]):
            df[f"sma_{period}"] = self.calculate_sma(df["close"], period)

    def _add_macd(self, df: pd.DataFrame) -> None:
        macd_df = self.calculate_macd(df["close"])
        for col in macd_df.columns:
            df[col] = macd_df[col]

    def _add_adx(self, df: pd.DataFrame) -> None:
        adx_df = self.calculate_adx(df)
        for col in adx_df.columns:
            df[col] = adx_df[col]

    def _add_ichimoku(self, df: pd.DataFrame) -> None:
        ichi_df = self.calculate_ichimoku(df)
        for col in ichi_df.columns:
            df[col] = ichi_df[col]

        close_safe = df["close"].replace(0, np.nan)
        df["ichi_tk_diff"] = (
            (df["ichi_tenkan"] - df["ichi_kijun"]) / close_safe * 100.0
        )
        df["ichi_price_vs_kijun"] = (
            (df["close"] - df["ichi_kijun"]) / close_safe * 100.0
        )

    # ── Momentum ──

    def _add_rsi(self, df: pd.DataFrame) -> None:
        for period in self.config.get("rsi_periods", [14]):
            df[f"rsi_{period}"] = self.calculate_rsi(df["close"], period)

    def _add_stochastic(self, df: pd.DataFrame) -> None:
        stoch_df = self.calculate_stochastic(df)
        for col in stoch_df.columns:
            df[col] = stoch_df[col]

    def _add_williams_r(self, df: pd.DataFrame) -> None:
        df["williams_r"] = self.calculate_williams_r(df)

    def _add_cci(self, df: pd.DataFrame) -> None:
        period = self.config.get("cci_period", 20)
        df[f"cci_{period}"] = self.calculate_cci(df, period)

    def _add_mfi(self, df: pd.DataFrame) -> None:
        period = self.config.get("mfi_period", 14)
        df[f"mfi_{period}"] = self.calculate_mfi(df, period)

    # ── Volatility ──

    def _add_bollinger_bands(self, df: pd.DataFrame) -> None:
        bb_df = self.calculate_bollinger_bands(df)
        for col in bb_df.columns:
            df[col] = bb_df[col]

    def _add_atr(self, df: pd.DataFrame) -> None:
        period = self.config.get("atr_period", 14)
        atr = self.calculate_atr(df, period)
        df[f"atr_{period}"] = atr
        df["atr_pct"] = atr / df["close"].replace(0, np.nan) * 100.0

    # ── Volume ──

    def _add_obv(self, df: pd.DataFrame) -> None:
        df["obv"] = self.calculate_obv(df)
        df["obv_ema"] = self.calculate_ema(df["obv"], 20)

    def _add_vwap(self, df: pd.DataFrame) -> None:
        df["vwap"] = self.calculate_vwap(df)
        df["vwap_deviation"] = (
            (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan) * 100.0
        )

    def _add_volume_features(self, df: pd.DataFrame) -> None:
        for period in self.config.get("volume_ma_periods", [20]):
            vol_ma = df["volume"].rolling(
                window=period, min_periods=period
            ).mean()
            df[f"vol_ma_{period}"] = vol_ma
            df[f"vol_ratio_{period}"] = df["volume"] / vol_ma.replace(0, np.nan)

        if "taker_buy_volume" in df.columns:
            df["taker_buy_ratio"] = df["taker_buy_volume"] / df["volume"].replace(
                0, np.nan
            )

    # ── Statistical ──

    def _add_returns(self, df: pd.DataFrame) -> None:
        for period in self.config.get("returns_periods", [1, 5, 10]):
            df[f"return_{period}"] = df["close"].pct_change(period) * 100.0

    def _add_volatility(self, df: pd.DataFrame) -> None:
        pct_change = df["close"].pct_change()
        for period in self.config.get("volatility_periods", [20]):
            df[f"volatility_{period}"] = (
                pct_change.rolling(window=period, min_periods=period).std() * 100.0
            )

    def _add_lags(self, df: pd.DataFrame) -> None:
        pct = df["close"].pct_change() * 100.0
        for lag in self.config.get("lag_periods", [1, 2, 3]):
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
            df[f"return_lag_{lag}"] = pct.shift(lag)

    def _add_z_score(self, df: pd.DataFrame) -> None:
        period = self.config.get("z_score_period", 20)
        rolling_mean = df["close"].rolling(window=period, min_periods=period).mean()
        rolling_std = df["close"].rolling(window=period, min_periods=period).std()
        df["z_score"] = (df["close"] - rolling_mean) / rolling_std.replace(
            0, np.nan
        )

    # ── Derived / Composite ──

    def _add_price_vs_ma(self, df: pd.DataFrame) -> None:
        for period in self.config.get("ema_periods", []):
            col = f"ema_{period}"
            if col in df.columns:
                df[f"price_vs_ema_{period}"] = (
                    (df["close"] - df[col]) / df[col].replace(0, np.nan) * 100.0
                )

        for period in self.config.get("sma_periods", []):
            col = f"sma_{period}"
            if col in df.columns:
                df[f"price_vs_sma_{period}"] = (
                    (df["close"] - df[col]) / df[col].replace(0, np.nan) * 100.0
                )

    def _add_ma_slopes(self, df: pd.DataFrame) -> None:
        for period in self.config.get("ema_periods", []):
            col = f"ema_{period}"
            if col in df.columns:
                df[f"ema_slope_{period}"] = df[col].pct_change(3) * 100.0

    def _add_crossover_signals(self, df: pd.DataFrame) -> None:
        ema_periods = sorted(self.config.get("ema_periods", []))
        for i in range(len(ema_periods) - 1):
            fp, sp = ema_periods[i], ema_periods[i + 1]
            fc, sc = f"ema_{fp}", f"ema_{sp}"
            if fc in df.columns and sc in df.columns:
                df[f"ema_cross_{fp}_{sp}"] = (
                    (df[fc] - df[sc]) / df[sc].replace(0, np.nan) * 100.0
                )

        if "macd" in df.columns and "macd_signal" in df.columns:
            df["macd_cross"] = np.sign(df["macd"] - df["macd_signal"])

        if "stoch_k" in df.columns and "stoch_d" in df.columns:
            df["stoch_cross"] = np.sign(df["stoch_k"] - df["stoch_d"])

        if "rsi_14" in df.columns:
            df["rsi_14_vs_50"] = df["rsi_14"] - 50.0

    def _add_candle_features(self, df: pd.DataFrame) -> None:
        body = df["close"] - df["open"]
        full_range = (df["high"] - df["low"]).replace(0, np.nan)

        upper_body = df[["open", "close"]].max(axis=1)
        lower_body = df[["open", "close"]].min(axis=1)

        df["candle_body_pct"] = body / full_range * 100.0
        df["candle_upper_shadow"] = (df["high"] - upper_body) / full_range * 100.0
        df["candle_lower_shadow"] = (lower_body - df["low"]) / full_range * 100.0
        df["candle_direction"] = np.sign(body)

        direction = np.sign(body).fillna(0)
        groups = (direction != direction.shift(1)).cumsum()
        streak = direction.groupby(groups).cumcount() + 1
        df["consecutive_candles"] = streak * direction

    def _add_momentum_features(self, df: pd.DataFrame) -> None:
        if "rsi_14" in df.columns:
            df["rsi_14_slope"] = df["rsi_14"].diff(3)

        adx_col = f"adx_{self.config.get('adx_period', 14)}"
        if adx_col in df.columns:
            df["strong_trend"] = (df[adx_col] > 25).astype(int)

        if "plus_di" in df.columns and "minus_di" in df.columns:
            df["di_diff"] = df["plus_di"] - df["minus_di"]

        df["roc_10"] = df["close"].pct_change(10) * 100.0
        df["roc_20"] = df["close"].pct_change(20) * 100.0

        close_safe = df["close"].replace(0, np.nan)
        df["hl_pct"] = (df["high"] - df["low"]) / close_safe * 100.0

        df["dist_from_high_20"] = (
            (df["close"] - df["high"].rolling(20, min_periods=1).max())
            / close_safe
            * 100.0
        )
        df["dist_from_low_20"] = (
            (df["close"] - df["low"].rolling(20, min_periods=1).min())
            / close_safe
            * 100.0
        )

    # ══════════════════════════════════════════════
    #  UTILITY
    # ══════════════════════════════════════════════

    def get_feature_names(self) -> List[str]:
        """Return list of all generated indicator column names."""
        return self._feature_names.copy()

    def get_indicator_count(self) -> int:
        """Return total number of generated indicator columns."""
        return len(self._feature_names)


# ══════════════════════════════════════════════════
#  STANDALONE TEST
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  TECHNICAL ANALYZER — TEST")
    print("=" * 70)

    from data.binance_data import BinanceData

    # ── 1. Fetch real data ──
    bd = BinanceData()
    print("\n📡 Fetching BTCUSDT 1h (500 candles)...")
    df = bd.get_ohlcv("BTCUSDT", "1h", limit=500)

    if df is None or df.empty:
        print("❌ Failed to fetch data — check network / Binance")
        sys.exit(1)

    print(f"✅ Got {len(df)} candles")
    print(f"   Range: {df.index[0]}  →  {df.index[-1]}")

    # ── 2. Calculate all indicators ──
    ta = TechnicalAnalyzer()
    result = ta.calculate_all(df)

    features = ta.get_feature_names()
    print(f"\n📊 Results")
    print(f"   Original columns : {len(df.columns)}")
    print(f"   Total columns    : {len(result.columns)}")
    print(f"   Indicator columns: {ta.get_indicator_count()}")

    # ── 3. Group by prefix ──
    groups: Dict[str, List[str]] = {}
    for f in features:
        parts = f.split("_")
        if parts[0] in ("price", "vol", "dist", "candle", "close", "return",
                         "ema", "sma", "bb", "rsi", "ichi", "strong", "di"):
            prefix = "_".join(parts[:2])
        else:
            prefix = parts[0]
        groups.setdefault(prefix, []).append(f)

    print(f"\n📋 Features by group:")
    for grp in sorted(groups):
        cols = groups[grp]
        print(f"   {grp:>22s}  ({len(cols):>2d}): {', '.join(cols)}")

    # ── 4. NaN analysis ──
    nan_pct = result[features].isna().mean() * 100
    print(f"\n🔍 NaN Analysis ({len(features)} features, {len(result)} rows):")
    print(f"    0% NaN : {(nan_pct == 0).sum()}")
    print(f"   <10% NaN: {(nan_pct < 10).sum()}")
    print(f"   <30% NaN: {(nan_pct < 30).sum()}")
    print(f"   ≥30% NaN: {(nan_pct >= 30).sum()}")

    bad = nan_pct[nan_pct >= 30].sort_values(ascending=False)
    if len(bad):
        print(f"\n   ⚠️  High NaN (≥30%):")
        for fname, pct in bad.head(10).items():
            print(f"      {fname:>28s}: {pct:5.1f}%")

    # ── 5. Sample values (last candle) ──
    sample = [
        "rsi_14", "ema_21", "sma_50", "macd", "macd_histogram",
        "atr_14", "atr_pct", "adx_14", "plus_di", "minus_di",
        "bb_pct_b", "bb_width", "stoch_k", "stoch_d",
        "obv", "vwap", "vwap_deviation",
        "williams_r", "cci_20", "mfi_14",
        "return_1", "volatility_20", "z_score",
        "price_vs_ema_21", "ema_slope_21",
        "ema_cross_9_21", "macd_cross", "rsi_14_vs_50",
        "candle_body_pct", "consecutive_candles",
        "roc_10", "di_diff", "dist_from_high_20",
    ]
    last = result.iloc[-1]
    print(f"\n📈 Last candle ({result.index[-1]}):")
    for c in sample:
        if c in result.columns:
            v = last[c]
            if pd.notna(v):
                if abs(v) > 1_000_000:
                    print(f"   {c:>25s}: {v:>16,.0f}")
                else:
                    print(f"   {c:>25s}: {v:>16.4f}")
            else:
                print(f"   {c:>25s}: {'NaN':>16s}")

    # ── 6. Sanity checks ──
    print(f"\n🧪 Sanity checks:")
    ok = True

    checks = [
        ("rsi_14",    0, 100,  "RSI 14 in [0,100]"),
        ("stoch_k",   0, 100,  "Stoch K in [0,100]"),
        ("mfi_14",    0, 100,  "MFI 14 in [0,100]"),
        ("williams_r",-100, 0, "Williams R in [-100,0]"),
    ]
    for col, lo, hi, label in checks:
        v = last.get(col)
        if v is not None and pd.notna(v):
            passed = lo <= v <= hi
            print(f"   {label:30s}: {'✅' if passed else '❌'} ({v:.2f})")
            ok = ok and passed

    bb_b = last.get("bb_pct_b")
    if bb_b is not None and pd.notna(bb_b):
        passed = -1 <= bb_b <= 2
        print(f"   {'BB %B reasonable':30s}: {'✅' if passed else '❌'} ({bb_b:.4f})")
        ok = ok and passed

    count = ta.get_indicator_count()
    passed = count >= 80
    print(f"   {'Feature count ≥ 80':30s}: {'✅' if passed else '❌'} ({count})")
    ok = ok and passed

    print(f"\n{'✅ All checks passed!' if ok else '⚠️  Some checks failed'}")
    print("=" * 70)