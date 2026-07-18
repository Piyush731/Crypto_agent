"""Pre-registered V4-T9 broader-universe dual-horizon trend strategy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from features.causal_builder import atr


@dataclass(frozen=True)
class CostAwareDualTrendConfig:
    return_72h_weight: float = 0.40
    return_30d_weight: float = 0.60
    long_lookback_hours: int = 720
    rebalance_hours: int = 8
    retention_rank: int = 3
    risk_per_leg_pct: float = 0.25
    max_notional_per_leg_pct: float = 25.0
    stop_atr_multiple: float = 2.0
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 2.0
    half_spread_bps_per_side: float = 1.0
    funding_bps_per_8h: float = 1.0


class CostAwareDualTrend:
    strategy_id = "cost_aware_dual_trend_1h"
    version = 1
    candidate_symbols = (
        "BTC", "ETH", "SOL", "BNB", "XRP",
        "DOGE", "ADA", "LINK", "LTC", "AVAX",
    )

    def __init__(self, config: CostAwareDualTrendConfig | None = None):
        self.config = config or CostAwareDualTrendConfig()

    def metadata(self):
        return {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "description": (
                "V4-T9: every 8h rank a fixed broader OKX universe by "
                "volatility-adjusted 72h/30d completed-1h trend and retain "
                "incumbents inside a top/bottom-three rank buffer."
            ),
            "candidate_symbols": list(self.candidate_symbols),
            "config": asdict(self.config),
        }

    def score_symbol(self, frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        cfg = self.config
        close = frame["close"]
        returns_1h = close.pct_change()
        ret_72h = close.pct_change(72)
        ret_30d = close.pct_change(cfg.long_lookback_hours)
        volatility = returns_1h.rolling(
            cfg.long_lookback_hours,
            min_periods=cfg.long_lookback_hours,
        ).std()
        raw = cfg.return_72h_weight * ret_72h + cfg.return_30d_weight * ret_30d
        score = raw / volatility.clip(lower=1e-6)
        atr_pct = atr(frame, 14) / close.replace(0, np.nan) * 100

        # An hourly candle labelled by its open is available one hour later.
        return pd.DataFrame(
            {
                f"{symbol}_score": score,
                f"{symbol}_raw": raw,
                f"{symbol}_atr_pct": atr_pct,
            },
            index=pd.to_datetime(
                frame.index.asi8 + 3_600_000_000_000,
                unit="ns",
                utc=True,
            ),
        )

    def select_with_buffer(
        self,
        scores: dict[str, float],
        raw: dict[str, float],
        incumbent_long: str | None,
        incumbent_short: str | None,
    ) -> tuple[str | None, str | None]:
        ranked = sorted(scores, key=lambda symbol: scores[symbol], reverse=True)
        retention = self.config.retention_rank

        top = ranked[0]
        bottom = ranked[-1]
        top_buffer = set(ranked[:retention])
        bottom_buffer = set(ranked[-retention:])

        if incumbent_long in top_buffer and raw.get(incumbent_long, 0.0) > 0:
            long_symbol = incumbent_long
        else:
            long_symbol = top if raw[top] > 0 else None

        if incumbent_short in bottom_buffer and raw.get(incumbent_short, 0.0) < 0:
            short_symbol = incumbent_short
        else:
            short_symbol = bottom if raw[bottom] < 0 else None

        if long_symbol == short_symbol:
            return None, None
        return long_symbol, short_symbol

    def build_schedule(self, hourly_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if len(hourly_frames) < 8:
            raise ValueError("V4-T9 requires at least 8 eligible symbols")

        combined = None
        for symbol, frame in sorted(hourly_frames.items()):
            scored = self.score_symbol(frame.sort_index(), symbol)
            combined = scored if combined is None else combined.join(scored, how="inner")

        combined = combined.dropna().sort_index()
        combined = combined[
            (combined.index.hour % self.config.rebalance_hours == 0)
            & (combined.index.minute == 0)
        ]

        symbols = sorted(hourly_frames)
        rows = []
        incumbent_long = None
        incumbent_short = None

        for timestamp, row in combined.iterrows():
            scores = {symbol: float(row[f"{symbol}_score"]) for symbol in symbols}
            raw = {symbol: float(row[f"{symbol}_raw"]) for symbol in symbols}
            long_symbol, short_symbol = self.select_with_buffer(
                scores, raw, incumbent_long, incumbent_short
            )
            incumbent_long = long_symbol
            incumbent_short = short_symbol

            rows.append(
                {
                    "timestamp": timestamp,
                    "long_symbol": long_symbol,
                    "short_symbol": short_symbol,
                    "long_score": scores[long_symbol] if long_symbol else None,
                    "short_score": scores[short_symbol] if short_symbol else None,
                    "score_spread": (
                        scores[long_symbol] - scores[short_symbol]
                        if long_symbol and short_symbol else None
                    ),
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
