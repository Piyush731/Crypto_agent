"""Pre-registered V4-T11 multi-horizon time-series momentum strategy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from features.causal_builder import atr

HOUR_NS = 3_600_000_000_000


@dataclass(frozen=True)
class TimeSeriesMomentumConfig:
    horizon_7d_hours: int = 168
    horizon_30d_hours: int = 720
    horizon_90d_hours: int = 2160
    volatility_lookback_hours: int = 720
    rebalance_weekday: int = 0  # Monday UTC
    rebalance_hour: int = 0
    maximum_positions: int = 6
    risk_per_leg_pct: float = 0.10
    max_notional_per_leg_pct: float = 10.0
    max_gross_notional_pct: float = 60.0
    stop_atr_multiple: float = 3.0
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 2.0
    half_spread_bps_per_side: float = 1.0
    funding_bps_per_8h: float = 1.0


class TimeSeriesMomentum:
    strategy_id = "time_series_momentum_1h"
    version = 1
    engine_type = "time_series_multi_leg_v1"
    candidate_symbols = (
        "BTC", "ETH", "SOL", "XRP", "DOGE",
        "ADA", "LINK", "LTC", "AVAX",
    )

    def __init__(self, config: TimeSeriesMomentumConfig | None = None):
        self.config = config or TimeSeriesMomentumConfig()

    def metadata(self):
        return {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "engine_type": self.engine_type,
            "description": (
                "V4-T11: weekly independent 7d/30d/90d trend voting, select "
                "up to six strongest absolute volatility-standardized signals."
            ),
            "candidate_symbols": list(self.candidate_symbols),
            "config": asdict(self.config),
        }

    def score_symbol(self, frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        cfg = self.config
        close = frame["close"]
        hourly_return = close.pct_change()
        ret_7d = close.pct_change(cfg.horizon_7d_hours)
        ret_30d = close.pct_change(cfg.horizon_30d_hours)
        ret_90d = close.pct_change(cfg.horizon_90d_hours)
        hourly_vol = hourly_return.rolling(
            cfg.volatility_lookback_hours,
            min_periods=cfg.volatility_lookback_hours,
        ).std().clip(lower=1e-6)
        standardized = (
            ret_7d / (hourly_vol * np.sqrt(cfg.horizon_7d_hours))
            + ret_30d / (hourly_vol * np.sqrt(cfg.horizon_30d_hours))
            + ret_90d / (hourly_vol * np.sqrt(cfg.horizon_90d_hours))
        ) / 3.0
        positive_votes = (
            (ret_7d > 0).astype(int)
            + (ret_30d > 0).astype(int)
            + (ret_90d > 0).astype(int)
        )
        negative_votes = (
            (ret_7d < 0).astype(int)
            + (ret_30d < 0).astype(int)
            + (ret_90d < 0).astype(int)
        )
        direction = pd.Series(
            np.where(
                positive_votes >= 2,
                1,
                np.where(negative_votes >= 2, -1, 0),
            ),
            index=frame.index,
        )
        atr_pct = atr(frame, 14) / close.replace(0, np.nan) * 100
        available_index = pd.to_datetime(
            frame.index.asi8 + HOUR_NS,
            unit="ns",
            utc=True,
        )
        return pd.DataFrame(
            {
                f"{symbol}_direction": direction,
                f"{symbol}_strength": standardized.abs(),
                f"{symbol}_signed_score": standardized,
                f"{symbol}_atr_pct": atr_pct,
            },
            index=available_index,
        )

    def build_schedule(self, hourly_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if set(hourly_frames) != set(self.candidate_symbols):
            raise ValueError(
                "V4-T11 requires exactly the nine coverage-eligible symbols"
            )
        combined = None
        for symbol, frame in sorted(hourly_frames.items()):
            scored = self.score_symbol(frame.sort_index(), symbol)
            combined = scored if combined is None else combined.join(scored, how="inner")
        combined = combined.dropna().sort_index()
        combined = combined[
            (combined.index.weekday == self.config.rebalance_weekday)
            & (combined.index.hour == self.config.rebalance_hour)
            & (combined.index.minute == 0)
        ]

        symbols = sorted(hourly_frames)
        rows = []
        for timestamp, row in combined.iterrows():
            candidates = []
            for symbol in symbols:
                direction = int(row[f"{symbol}_direction"])
                if direction == 0:
                    continue
                candidates.append(
                    (
                        symbol,
                        direction,
                        float(row[f"{symbol}_strength"]),
                        float(row[f"{symbol}_signed_score"]),
                        float(row[f"{symbol}_atr_pct"]),
                    )
                )
            selected = sorted(
                candidates,
                key=lambda item: (-item[2], item[0]),
            )[: self.config.maximum_positions]
            longs = tuple(item[0] for item in selected if item[1] == 1)
            shorts = tuple(item[0] for item in selected if item[1] == -1)
            rows.append(
                {
                    "timestamp": timestamp,
                    "long_symbols": longs,
                    "short_symbols": shorts,
                    "atr_pct": {item[0]: item[4] for item in selected},
                    "signed_score": {item[0]: item[3] for item in selected},
                }
            )
        return pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()
