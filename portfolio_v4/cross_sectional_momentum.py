"""Pre-registered market-neutral cross-sectional momentum strategy (V4-T7)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from features.causal_builder import atr


@dataclass(frozen=True)
class CrossSectionalMomentumConfig:
    return_24h_weight: float = 0.40
    return_7d_weight: float = 0.60
    volatility_lookback_hours: int = 168
    rebalance_hours: int = 4
    risk_per_leg_pct: float = 0.25
    max_notional_per_leg_pct: float = 25.0
    stop_atr_multiple: float = 2.0
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 2.0
    half_spread_bps_per_side: float = 1.0
    funding_bps_per_8h: float = 1.0


class CrossSectionalMomentum:
    strategy_id = "cross_sectional_momentum_1h"
    version = 1

    def __init__(self, config: CrossSectionalMomentumConfig | None = None):
        self.config = config or CrossSectionalMomentumConfig()

    def metadata(self):
        return {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "description": (
                "Every 4h, long the strongest and short the weakest OKX swap "
                "by volatility-adjusted 24h/7d completed-1h momentum."
            ),
            "config": asdict(self.config),
        }

    def score_symbol(self, frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        cfg = self.config
        close = frame["close"]
        returns_1h = close.pct_change()
        ret_24h = close.pct_change(24)
        ret_7d = close.pct_change(168)
        volatility = returns_1h.rolling(
            cfg.volatility_lookback_hours,
            min_periods=cfg.volatility_lookback_hours,
        ).std()
        raw_momentum = (
            cfg.return_24h_weight * ret_24h
            + cfg.return_7d_weight * ret_7d
        )
        score = raw_momentum / volatility.clip(lower=1e-6)
        atr_pct = atr(frame, 14) / close.replace(0, np.nan) * 100

        # A completed 1h candle becomes available one hour after its open.
        result = pd.DataFrame(
            {
                f"{symbol}_score": score,
                f"{symbol}_raw": raw_momentum,
                f"{symbol}_atr_pct": atr_pct,
            },
            index=frame.index + pd.to_timedelta(1, unit="h"),
        )
        return result

    def build_schedule(self, hourly_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if len(hourly_frames) < 3:
            raise ValueError("Cross-sectional strategy needs at least 3 symbols")

        combined = None
        for symbol, frame in sorted(hourly_frames.items()):
            scored = self.score_symbol(frame.sort_index(), symbol)
            combined = scored if combined is None else combined.join(scored, how="inner")

        combined = combined.dropna().sort_index()
        combined = combined[
            (combined.index.hour % self.config.rebalance_hours == 0)
            & (combined.index.minute == 0)
        ]

        rows = []
        symbols = sorted(hourly_frames)
        for timestamp, row in combined.iterrows():
            scores = {symbol: float(row[f"{symbol}_score"]) for symbol in symbols}
            raw = {symbol: float(row[f"{symbol}_raw"]) for symbol in symbols}
            strongest = max(symbols, key=lambda symbol: scores[symbol])
            weakest = min(symbols, key=lambda symbol: scores[symbol])

            # Stay in cash unless the universe contains both positive and
            # negative raw momentum. This avoids forcing a market-neutral pair.
            long_symbol = strongest if raw[strongest] > 0 else None
            short_symbol = weakest if raw[weakest] < 0 else None
            if long_symbol == short_symbol:
                long_symbol = None
                short_symbol = None

            rows.append(
                {
                    "timestamp": timestamp,
                    "long_symbol": long_symbol,
                    "short_symbol": short_symbol,
                    "long_score": scores[strongest],
                    "short_score": scores[weakest],
                    "score_spread": scores[strongest] - scores[weakest],
                    "long_atr_pct": (
                        float(row[f"{long_symbol}_atr_pct"])
                        if long_symbol else None
                    ),
                    "short_atr_pct": (
                        float(row[f"{short_symbol}_atr_pct"])
                        if short_symbol else None
                    ),
                }
            )

        return pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()
