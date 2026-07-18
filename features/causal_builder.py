"""Leakage-resistant 5m/15m/1h feature builder for OKX v4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.market_store import MarketStore


@dataclass(frozen=True)
class CausalFeatureConfig:
    horizon_bars: int = 48          # 4 hours on 5m bars
    neutral_return: float = 0.003   # +/-0.30% => HOLD class
    base_interval: pd.Timedelta = pd.to_timedelta(5, unit="min")
    setup_interval: pd.Timedelta = pd.to_timedelta(15, unit="min")
    trend_interval: pd.Timedelta = pd.to_timedelta(1, unit="h")


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return result


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def macd_histogram(series: pd.Series) -> pd.Series:
    fast = series.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = series.ewm(span=26, adjust=False, min_periods=26).mean()
    line = fast - slow
    signal = line.ewm(span=9, adjust=False, min_periods=9).mean()
    return line - signal


def adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - frame["close"].shift(1)).abs(),
            (frame["low"] - frame["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    smoothed_tr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean() / smoothed_tr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean() / smoothed_tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def timeframe_features(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    close = frame["close"]
    result = pd.DataFrame(index=frame.index)
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    atr14 = atr(frame, 14)
    result[f"{prefix}_rsi14"] = rsi(close, 14)
    result[f"{prefix}_ema21_dist"] = (close / ema21 - 1) * 100
    result[f"{prefix}_ema50_dist"] = (close / ema50 - 1) * 100
    result[f"{prefix}_ema21_slope3"] = ema21.pct_change(3) * 100
    result[f"{prefix}_macd_hist"] = macd_histogram(close)
    result[f"{prefix}_atr_pct"] = atr14 / close.replace(0, np.nan) * 100
    result[f"{prefix}_adx14"] = adx(frame, 14)
    prior_high_20 = frame["high"].shift(1).rolling(20, min_periods=20).max()
    prior_low_20 = frame["low"].shift(1).rolling(20, min_periods=20).min()
    result[f"{prefix}_breakout_high20_pct"] = (
        close / prior_high_20.replace(0, np.nan) - 1
    ) * 100
    result[f"{prefix}_breakout_low20_pct"] = (
        close / prior_low_20.replace(0, np.nan) - 1
    ) * 100
    prior_high_55 = frame["high"].shift(1).rolling(55, min_periods=55).max()
    prior_low_55 = frame["low"].shift(1).rolling(55, min_periods=55).min()
    result[f"{prefix}_breakout_high55_pct"] = (
        close / prior_high_55.replace(0, np.nan) - 1
    ) * 100
    result[f"{prefix}_breakout_low55_pct"] = (
        close / prior_low_55.replace(0, np.nan) - 1
    ) * 100
    result[f"{prefix}_volume_ratio20"] = frame["volume"] / frame["volume"].rolling(
        20, min_periods=20
    ).mean().replace(0, np.nan)
    middle = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std()
    upper = middle + 2 * std
    lower = middle - 2 * std
    width = (upper - lower).replace(0, np.nan)
    result[f"{prefix}_bb_pct_b"] = (close - lower) / width
    return result


def base_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"]
    result = pd.DataFrame(index=frame.index)

    for bars in (1, 3, 6, 12, 24, 48):
        result[f"ret_{bars}"] = close.pct_change(bars) * 100
    for period in (9, 21, 50, 200):
        line = ema(close, period)
        result[f"ema{period}_dist"] = (close / line - 1) * 100
        if period in (21, 50):
            result[f"ema{period}_slope3"] = line.pct_change(3) * 100

    result["rsi7"] = rsi(close, 7)
    result["rsi14"] = rsi(close, 14)
    result["macd_hist"] = macd_histogram(close)
    atr14 = atr(frame, 14)
    result["atr_pct"] = atr14 / close.replace(0, np.nan) * 100
    result["adx14"] = adx(frame, 14)

    middle = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std()
    upper = middle + 2 * std
    lower = middle - 2 * std
    width = (upper - lower).replace(0, np.nan)
    result["bb_pct_b"] = (close - lower) / width
    result["bb_width_pct"] = width / middle.replace(0, np.nan) * 100

    result["vol_ratio_12"] = frame["volume"] / frame["volume"].rolling(
        12, min_periods=12
    ).mean().replace(0, np.nan)
    result["vol_ratio_48"] = frame["volume"] / frame["volume"].rolling(
        48, min_periods=48
    ).mean().replace(0, np.nan)
    returns = close.pct_change()
    result["volatility_12"] = returns.rolling(12, min_periods=12).std() * 100
    result["volatility_48"] = returns.rolling(48, min_periods=48).std() * 100

    candle_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    result["body_pct"] = (frame["close"] - frame["open"]) / candle_range * 100
    result["range_pct"] = candle_range / close.replace(0, np.nan) * 100
    result["close_location"] = (close - frame["low"]) / candle_range

    decision_index = frame.index + pd.to_timedelta(5, unit="min")
    result.index = decision_index
    result["hour_sin"] = np.sin(2 * np.pi * result.index.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result.index.hour / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result.index.dayofweek / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result.index.dayofweek / 7)
    return result


def available_higher_features(
    frame: pd.DataFrame,
    prefix: str,
    availability_delay: pd.Timedelta,
) -> pd.DataFrame:
    features = timeframe_features(frame, prefix)
    features.index = features.index + availability_delay
    return features.sort_index()


def backward_asof_join(base: pd.DataFrame, higher: pd.DataFrame) -> pd.DataFrame:
    left = base.sort_index().reset_index(names="decision_time")
    right = higher.sort_index().reset_index(names="available_time")
    merged = pd.merge_asof(
        left,
        right,
        left_on="decision_time",
        right_on="available_time",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.drop(columns=["available_time"]).set_index("decision_time")


class CausalFeatureBuilder:
    def __init__(self, store: MarketStore, config: CausalFeatureConfig | None = None):
        self.store = store
        self.config = config or CausalFeatureConfig()

    def build(
        self,
        instrument_id: str,
        include_target: bool = True,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> dict:
        base = self.store.get_candles(
            "okx", instrument_id, "5m", start_ms, end_ms, completed_only=True
        )
        setup = self.store.get_candles(
            "okx", instrument_id, "15m", start_ms, end_ms, completed_only=True
        )
        trend = self.store.get_candles(
            "okx", instrument_id, "1h", start_ms, end_ms, completed_only=True
        )
        if base.empty or setup.empty or trend.empty:
            raise ValueError("5m/15m/1h completed data is required")

        features = base_features(base)
        features = backward_asof_join(
            features,
            available_higher_features(setup, "tf15", self.config.setup_interval),
        )
        features = backward_asof_join(
            features,
            available_higher_features(trend, "tf1h", self.config.trend_interval),
        )
        features = features.replace([np.inf, -np.inf], np.nan)

        prices = base[["open", "high", "low", "close", "volume"]].copy()
        prices.index = prices.index + self.config.base_interval
        prices = prices.reindex(features.index)

        target = None
        target_end_time = None
        future_return = None
        if include_target:
            future_close = base["close"].shift(-self.config.horizon_bars)
            future_return_open_index = future_close / base["close"] - 1
            future_end_open_index = pd.Series(
                base.index, index=base.index
            ).shift(-self.config.horizon_bars)

            future_return = future_return_open_index.copy()
            future_return.index = future_return.index + self.config.base_interval
            target_end_time = future_end_open_index + self.config.base_interval
            target_end_time.index = target_end_time.index + self.config.base_interval

            target = pd.Series(0, index=future_return.index, dtype="float64")
            target[future_return > self.config.neutral_return] = 1
            target[future_return < -self.config.neutral_return] = -1
            target[future_return.isna()] = np.nan

        valid = features.notna().all(axis=1)
        if include_target:
            valid &= target.notna() & target_end_time.notna()

        features = features.loc[valid].astype("float64")
        prices = prices.loc[features.index]
        if include_target:
            target = target.loc[features.index].astype("int8")
            target_end_time = pd.to_datetime(target_end_time.loc[features.index], utc=True)
            future_return = future_return.loc[features.index].astype("float64")

        return {
            "features": features,
            "target": target,
            "target_end_time": target_end_time,
            "future_return": future_return,
            "prices": prices,
            "metadata": {
                "instrument_id": instrument_id,
                "rows": len(features),
                "feature_count": features.shape[1],
                "horizon_bars": self.config.horizon_bars,
                "neutral_return": self.config.neutral_return,
                "first_decision": features.index.min(),
                "last_decision": features.index.max(),
            },
        }
