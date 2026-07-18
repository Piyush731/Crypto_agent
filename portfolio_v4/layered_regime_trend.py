"""Pre-registered V4-T10 layered, regime-aware cross-sectional trend strategy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from features.causal_builder import atr

HOUR_NS = 3_600_000_000_000


@dataclass(frozen=True)
class LayeredRegimeTrendConfig:
    return_7d_weight: float = 0.50
    return_30d_weight: float = 0.50
    long_lookback_hours: int = 720
    breadth_threshold: float = 0.60
    rebalance_hours: int = 12
    retention_rank: int = 4
    positions_per_regime: int = 2
    risk_per_leg_pct: float = 0.50
    max_notional_per_leg_pct: float = 30.0
    max_gross_notional_pct: float = 60.0
    stop_atr_multiple: float = 2.0
    trail_arm_atr_multiple: float = 2.0
    trail_distance_atr_multiple: float = 2.0
    half_risk_drawdown_pct: float = 7.5
    stop_new_drawdown_pct: float = 12.5
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 2.0
    half_spread_bps_per_side: float = 1.0
    funding_bps_per_8h: float = 1.0


class LayeredRegimeTrend:
    strategy_id = "layered_regime_trend_1h"
    version = 1
    engine_type = "layered_multi_leg_v1"
    candidate_symbols = (
        "BTC", "ETH", "SOL", "BNB", "XRP",
        "DOGE", "ADA", "LINK", "LTC", "AVAX",
    )

    def __init__(self, config: LayeredRegimeTrendConfig | None = None):
        self.config = config or LayeredRegimeTrendConfig()

    def metadata(self):
        return {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "engine_type": self.engine_type,
            "description": (
                "V4-T10: completed-1h 7d/30d cross-sectional trend, gated by "
                "BTC 30d EMA and 60% market breadth; two directional legs, "
                "12h rebalance, top/bottom-four retention."
            ),
            "candidate_symbols": list(self.candidate_symbols),
            "config": asdict(self.config),
        }

    def score_symbol(self, frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        cfg = self.config
        close = frame["close"]
        returns_1h = close.pct_change()
        ret_7d = close.pct_change(168)
        ret_30d = close.pct_change(cfg.long_lookback_hours)
        volatility = returns_1h.rolling(
            cfg.long_lookback_hours,
            min_periods=cfg.long_lookback_hours,
        ).std()
        raw = cfg.return_7d_weight * ret_7d + cfg.return_30d_weight * ret_30d
        score = raw / volatility.clip(lower=1e-6)
        atr_pct = atr(frame, 14) / close.replace(0, np.nan) * 100
        ema_30d = close.ewm(
            span=cfg.long_lookback_hours,
            adjust=False,
            min_periods=cfg.long_lookback_hours,
        ).mean()
        available_index = pd.to_datetime(
            frame.index.asi8 + HOUR_NS,
            unit="ns",
            utc=True,
        )
        return pd.DataFrame(
            {
                f"{symbol}_score": score,
                f"{symbol}_raw": raw,
                f"{symbol}_ret30d": ret_30d,
                f"{symbol}_atr_pct": atr_pct,
                f"{symbol}_close": close,
                f"{symbol}_ema30d": ema_30d,
            },
            index=available_index,
        )

    def regime(self, row: pd.Series, symbols: list[str]) -> str:
        positive_breadth = float(
            np.mean([float(row[f"{symbol}_ret30d"]) > 0 for symbol in symbols])
        )
        btc_above_ema = float(row["BTC_close"]) > float(row["BTC_ema30d"])
        if btc_above_ema and positive_breadth >= self.config.breadth_threshold:
            return "bull"
        if (not btc_above_ema) and positive_breadth <= (1 - self.config.breadth_threshold):
            return "bear"
        return "mixed"

    def select_with_retention(
        self,
        scores: dict[str, float],
        raw: dict[str, float],
        regime: str,
        incumbent_longs: tuple[str, ...],
        incumbent_shorts: tuple[str, ...],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        ranked = sorted(scores, key=lambda symbol: scores[symbol], reverse=True)
        keep_rank = self.config.retention_rank
        target_count = self.config.positions_per_regime

        if regime == "bull":
            retained = [
                symbol for symbol in incumbent_longs
                if symbol in ranked[:keep_rank] and raw[symbol] > 0
            ][:target_count]
            for symbol in ranked:
                if len(retained) >= target_count:
                    break
                if raw[symbol] > 0 and symbol not in retained:
                    retained.append(symbol)
            return tuple(retained), ()

        if regime == "bear":
            retained = [
                symbol for symbol in incumbent_shorts
                if symbol in ranked[-keep_rank:] and raw[symbol] < 0
            ][:target_count]
            for symbol in reversed(ranked):
                if len(retained) >= target_count:
                    break
                if raw[symbol] < 0 and symbol not in retained:
                    retained.append(symbol)
            return (), tuple(retained)

        return (), ()

    def build_schedule(self, hourly_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if len(hourly_frames) != len(self.candidate_symbols):
            raise ValueError(
                f"V4-T10 requires all {len(self.candidate_symbols)} pre-registered symbols"
            )
        missing = set(self.candidate_symbols) - set(hourly_frames)
        if missing:
            raise ValueError(f"Missing V4-T10 symbols: {sorted(missing)}")

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
        incumbent_longs: tuple[str, ...] = ()
        incumbent_shorts: tuple[str, ...] = ()
        rows = []

        for timestamp, row in combined.iterrows():
            scores = {symbol: float(row[f"{symbol}_score"]) for symbol in symbols}
            raw = {symbol: float(row[f"{symbol}_raw"]) for symbol in symbols}
            current_regime = self.regime(row, symbols)
            longs, shorts = self.select_with_retention(
                scores, raw, current_regime, incumbent_longs, incumbent_shorts
            )
            incumbent_longs, incumbent_shorts = longs, shorts
            atr_pct = {
                symbol: float(row[f"{symbol}_atr_pct"])
                for symbol in set(longs + shorts)
            }
            rows.append(
                {
                    "timestamp": timestamp,
                    "regime": current_regime,
                    "long_symbols": longs,
                    "short_symbols": shorts,
                    "atr_pct": atr_pct,
                    "positive_breadth": float(
                        np.mean([row[f"{symbol}_ret30d"] > 0 for symbol in symbols])
                    ),
                }
            )

        return pd.DataFrame(rows).set_index("timestamp") if rows else pd.DataFrame()
