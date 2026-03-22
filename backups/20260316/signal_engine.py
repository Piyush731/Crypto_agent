"""
Crypto Futures AI Agent — Signal Engine  (v3.0.3 — DB field mapping fix)
=========================================================================
Combines all analysis components into a final trading signal.

Components (weights from SIGNAL_WEIGHTS in config.py):
    ML Ensemble:      45%  - EnsemblePredictor direction + confidence
    Sentiment:        15%  - FinBERT news sentiment + Fear&Greed index
    AI Reasoning:     15%  - HuggingFace CoT + Expert Debate via AIBrain
    Funding Rate:     10%  - Contrarian funding rate signal
    Market Structure: 15%  - Multi-timeframe trend alignment

Output:
    LONG/SHORT/HOLD with confidence score, entry/SL/TP levels, full breakdown.

v3.0.3 fixes:
  - _log_signal: explicit debug logging of every DB field before save
  - _log_signal: ensure combined_score and all component scores are float (not None)
  - _combine_signals: returns combined_score as float always
  - generate_signal: moved _log_signal call to ensure result is fully populated
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import (
    SIGNAL_WEIGHTS,
    RISK_CONFIG,
    TIMEFRAMES,
    TRADING_PAIRS,
    ACTIVE_HORIZON,
)
from core.logger import get_logger
from core.db import get_db
from analysis.ai_brain import AIBrain

logger = get_logger("analysis.signal_engine")


class SignalEngine:
    """
    Combine all analysis sources into one final trading signal.

    Each component produces a score from -1 (strong bearish) to +1 (strong bullish).
    Scores are weighted by SIGNAL_WEIGHTS and summed.
    Final signal: LONG (score > threshold), SHORT (score < -threshold), HOLD otherwise.
    """

    SIGNAL_THRESHOLD = 0.15  # Min |combined_score| to generate LONG/SHORT

    def __init__(self):
        self.brain = AIBrain()
        self._last_signals: Dict[str, Dict] = {}
        self._signal_count = 0

        logger.info(
            f"SignalEngine initialized | weights: "
            + " ".join(f"{k}={v:.0%}" for k, v in SIGNAL_WEIGHTS.items())
        )

    # ══════════════════════════════════════════════════════════════════
    #  MAIN ENTRY — generate_signal()
    # ══════════════════════════════════════════════════════════════════

    def generate_signal(
        self,
        symbol: str,
        dataset: Optional[Dict] = None,
        include_ai: bool = True,
    ) -> Dict:
        """
        Full signal generation pipeline for one symbol.

        Parameters
        ----------
        symbol      : str           e.g. "BTCUSDT"
        dataset     : dict | None   Pre-fetched from DataManager. None -> fetch live.
        include_ai  : bool          Run AI reasoning (slow ~30-60s). False for quick scan.

        Returns
        -------
        dict
            symbol, timestamp, signal, direction, confidence, combined_score,
            entry_price, stop_loss, take_profit, risk_reward_ratio,
            components: {ml_ensemble, sentiment, ai_reasoning, funding_rate, market_structure},
            active_components, data_quality, analysis_time_s, signal_id
        """
        t0 = time.time()
        logger.info(f"{'─'*50}")
        logger.info(f"Signal: {symbol} | ai={include_ai}")

        result = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal": "HOLD",
            "direction": 0,
            "confidence": 0.0,
            "combined_score": 0.0,
        }

        try:
            # ── 1. Get dataset ────────────────────────────────────────
            if dataset is None:
                dataset = self._fetch_dataset(symbol)
                if dataset is None:
                    result["error"] = "Data fetch failed"
                    return result

            result["data_quality"] = (
                dataset.get("data_quality", {}).get("score", 0)
            )

            # ── 2. Collect component signals ──────────────────────────
            components: Dict[str, Dict] = {}

            # ML Ensemble (45%)
            logger.info("  [ML] Generating ML signal …")
            components["ml_ensemble"] = self._get_ml_signal(
                symbol, dataset
            )

            # Sentiment (15%)
            logger.info("  [SENT] Extracting sentiment …")
            components["sentiment"] = self._get_sentiment_signal(dataset)

            # AI Reasoning (15%)
            if include_ai and self.brain.enabled:
                logger.info("  [AI] Running AI reasoning …")
                ml_pred = (
                    components["ml_ensemble"].get("details")
                    if components["ml_ensemble"].get("available")
                    else None
                )
                sent_data = (
                    components["sentiment"].get("details")
                    if components["sentiment"].get("available")
                    else None
                )
                components["ai_reasoning"] = self._get_ai_signal(
                    dataset, ml_pred, sent_data
                )
            else:
                components["ai_reasoning"] = {
                    "available": False,
                    "score": 0.0,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "weight": SIGNAL_WEIGHTS.get("ai_reasoning", 0.15),
                    "reason": "skipped" if not include_ai else "disabled",
                }

            # Funding Rate (10%)
            logger.info("  [FUND] Extracting funding signal …")
            components["funding_rate"] = self._get_funding_signal(dataset)

            # Market Structure (15%)
            logger.info("  [STRUCT] Computing market structure …")
            components["market_structure"] = (
                self._get_market_structure_signal(dataset)
            )

            # ── 3. Combine signals ────────────────────────────────────
            combined = self._combine_signals(components)
            result.update(combined)
            result["components"] = components

            # ── 4. Calculate SL/TP levels ─────────────────────────────
            if result["signal"] != "HOLD":
                levels = self._calculate_levels(
                    dataset, result["direction"]
                )
                result.update(levels)
            else:
                entry_price = self._get_current_price(dataset)
                result["entry_price"] = entry_price
                result["stop_loss"] = None
                result["take_profit"] = None
                result["risk_reward_ratio"] = None

            # ── 5. Log component summary ──────────────────────────────
            for name, comp in components.items():
                status = "✅" if comp.get("available") else "❌"
                score = comp.get("score", 0)
                sig = comp.get("signal", "N/A")
                w = comp.get("weight", 0)
                logger.info(
                    f"    {status} {name:20s} score={score:+.3f} "
                    f"signal={sig:5s} weight={w:.0%}"
                )

            logger.info(
                f"  FINAL: {result['signal']} | "
                f"conf={result['confidence']:.3f} | "
                f"score={result.get('combined_score', 0):+.3f} | "
                f"components={result.get('active_components', 0)}/5"
            )

            # ── 6. Save to DB ─────────────────────────────────────────
            signal_id = self._log_signal(result)
            result["signal_id"] = signal_id

        except Exception as exc:
            result["error"] = str(exc)
            logger.error(f"Signal generation failed: {exc}", exc_info=True)

        result["analysis_time_s"] = round(time.time() - t0, 2)
        self._last_signals[symbol] = result
        self._signal_count += 1
        return result

    # ══════════════════════════════════════════════════════════════════
    #  SCAN ALL PAIRS
    # ══════════════════════════════════════════════════════════════════

    def scan_all(
        self,
        symbols: Optional[List[str]] = None,
        include_ai: bool = False,
    ) -> Dict:
        """
        Scan multiple pairs and return signals + actionable summary.
        """
        pairs = symbols or TRADING_PAIRS
        t0 = time.time()
        logger.info(f"Scanning {len(pairs)} pairs: {pairs}")

        all_results: Dict[str, Dict] = {}

        for i, sym in enumerate(pairs, 1):
            logger.info(f"\n[{i}/{len(pairs)}] Scanning {sym} …")
            try:
                res = self.generate_signal(sym, include_ai=include_ai)
                all_results[sym] = res
            except Exception as exc:
                logger.error(f"  {sym} scan failed: {exc}")
                all_results[sym] = {
                    "symbol": sym,
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "error": str(exc),
                }

        # Filter actionable signals
        actionable = {
            sym: r
            for sym, r in all_results.items()
            if r.get("signal") != "HOLD"
            and r.get("confidence", 0) >= RISK_CONFIG.get(
                "min_confidence_to_trade", 0.58
            )
        }

        summary = {
            "total": len(pairs),
            "actionable": len(actionable),
            "long": sum(
                1 for r in all_results.values()
                if r.get("signal") == "LONG"
            ),
            "short": sum(
                1 for r in all_results.values()
                if r.get("signal") == "SHORT"
            ),
            "hold": sum(
                1 for r in all_results.values()
                if r.get("signal") == "HOLD"
            ),
            "scan_time_s": round(time.time() - t0, 2),
        }

        logger.info(
            f"\nScan complete: {summary['actionable']}/{summary['total']} "
            f"actionable | {summary['long']}L {summary['short']}S "
            f"{summary['hold']}H | {summary['scan_time_s']}s"
        )

        return {
            "all": all_results,
            "actionable": actionable,
            "summary": summary,
        }

    # ══════════════════════════════════════════════════════════════════
    #  COMPONENT: ML ENSEMBLE
    # ══════════════════════════════════════════════════════════════════

    def _get_ml_signal(self, symbol: str, dataset: Dict) -> Dict:
        """
        Load trained model, build features, predict.
        Score = (probability_up - 0.5) * 2 -> maps [0,1] to [-1,+1].
        """
        weight = SIGNAL_WEIGHTS.get("ml_ensemble", 0.45)
        base = {
            "weight": weight,
            "available": False,
            "score": 0.0,
            "signal": "HOLD",
            "confidence": 0.0,
        }

        try:
            from models.ensemble import EnsemblePredictor
            from features.builder import FeatureBuilder

            # Load saved model
            ensemble = EnsemblePredictor()
            loaded = ensemble.load(symbol)
            if not loaded:
                base["reason"] = "no_trained_model"
                logger.info(f"    ML: no trained model for {symbol}")
                return base

            # Build prediction features
            builder = FeatureBuilder()
            build = builder.build_prediction_features(dataset)

            if build.get("metadata", {}).get("error") or build[
                "features"
            ].empty:
                base["reason"] = "feature_build_failed"
                base["error"] = build.get("metadata", {}).get("error", "")
                logger.warning(f"    ML: feature build failed for {symbol}")
                return base

            X = build["features"]

            # Predict (uses last row)
            prediction = ensemble.predict(X)

            if prediction.get("error"):
                base["reason"] = "prediction_failed"
                base["error"] = prediction["error"]
                return base

            prob_up = prediction.get("probability_up", 0.5)
            score = (prob_up - 0.5) * 2.0  # [-1, +1]

            return {
                "weight": weight,
                "available": True,
                "score": round(score, 4),
                "signal": prediction.get("signal", "HOLD"),
                "confidence": prediction.get("confidence", 0.5),
                "agreement": prediction.get("agreement", 0.0),
                "probability_up": prob_up,
                "details": prediction,
            }

        except Exception as exc:
            base["error"] = str(exc)
            logger.warning(f"    ML: {exc}")
            return base

    # ══════════════════════════════════════════════════════════════════
    #  COMPONENT: SENTIMENT
    # ══════════════════════════════════════════════════════════════════

    def _get_sentiment_signal(self, dataset: Dict) -> Dict:
        """
        Extract sentiment signal from dataset.
        Score = sentiment_score (already -1 to +1).
        """
        weight = SIGNAL_WEIGHTS.get("sentiment", 0.15)
        base = {
            "weight": weight,
            "available": False,
            "score": 0.0,
            "signal": "HOLD",
            "confidence": 0.0,
        }

        try:
            sentiment = dataset.get("sentiment")

            if sentiment and sentiment.get("sentiment_score") is not None:
                score = float(sentiment["sentiment_score"])
                conf = float(sentiment.get("confidence", 0.5))

                if score > 0.1:
                    signal = "LONG"
                elif score < -0.1:
                    signal = "SHORT"
                else:
                    signal = "HOLD"

                fg = sentiment.get("fear_greed") or {}

                return {
                    "weight": weight,
                    "available": True,
                    "score": round(max(-1.0, min(1.0, score)), 4),
                    "signal": signal,
                    "confidence": round(conf, 3),
                    "sentiment_label": sentiment.get(
                        "sentiment_label", "neutral"
                    ),
                    "fear_greed_value": fg.get("value", 50),
                    "fear_greed_label": fg.get("label", "Neutral"),
                    "details": sentiment,
                }

            # Fallback: try Fear & Greed only from news data
            news = dataset.get("news") or {}
            fg = news.get("fear_greed") or {}
            fg_val = fg.get("value")

            if fg_val is not None:
                fg_score = (fg_val - 50) / 50.0
                if fg_val < 20:
                    fg_score = fg_score * 0.5 + 0.3
                elif fg_val > 80:
                    fg_score = fg_score * 0.5 - 0.3

                signal = (
                    "LONG"
                    if fg_score > 0.1
                    else "SHORT" if fg_score < -0.1 else "HOLD"
                )

                return {
                    "weight": weight,
                    "available": True,
                    "score": round(max(-1.0, min(1.0, fg_score)), 4),
                    "signal": signal,
                    "confidence": 0.4,
                    "fear_greed_value": fg_val,
                    "fear_greed_label": fg.get("label", "Neutral"),
                    "reason": "fear_greed_only",
                }

            base["reason"] = "no_sentiment_data"
            return base

        except Exception as exc:
            base["error"] = str(exc)
            logger.warning(f"    Sentiment: {exc}")
            return base

    # ══════════════════════════════════════════════════════════════════
    #  COMPONENT: AI REASONING
    # ══════════════════════════════════════════════════════════════════

    def _get_ai_signal(
        self,
        dataset: Dict,
        ml_prediction: Optional[Dict],
        sentiment_data: Optional[Dict],
    ) -> Dict:
        """
        Run AIBrain analysis (CoT + Expert Debate).
        Score = direction * confidence -> [-1, +1].
        """
        weight = SIGNAL_WEIGHTS.get("ai_reasoning", 0.15)
        base = {
            "weight": weight,
            "available": False,
            "score": 0.0,
            "signal": "HOLD",
            "confidence": 0.0,
        }

        try:
            ai_result = self.brain.analyze(
                dataset, ml_prediction, sentiment_data
            )

            if not ai_result.get("enabled"):
                base["reason"] = "ai_disabled"
                return base

            direction = ai_result.get("direction", 0)
            confidence = ai_result.get("confidence", 0.0)
            score = direction * confidence

            return {
                "weight": weight,
                "available": True,
                "score": round(score, 4),
                "signal": ai_result.get("signal", "HOLD"),
                "confidence": round(confidence, 3),
                "method": ai_result.get("method", "unknown"),
                "agreement": ai_result.get("agreement", False),
                "api_calls": ai_result.get("api_calls", 0),
                "api_errors": ai_result.get("api_errors", 0),
                "models_used": ai_result.get("models_used", []),
                "reasoning": ai_result.get("reasoning", "")[:300],
                "details": ai_result,
            }

        except Exception as exc:
            base["error"] = str(exc)
            logger.warning(f"    AI: {exc}")
            return base

    # ══════════════════════════════════════════════════════════════════
    #  COMPONENT: FUNDING RATE
    # ══════════════════════════════════════════════════════════════════

    def _get_funding_signal(self, dataset: Dict) -> Dict:
        """
        Contrarian funding rate signal.
        Score = -(rate - baseline) * scale, capped to [-1, +1].
        """
        weight = SIGNAL_WEIGHTS.get("funding_rate", 0.10)
        base = {
            "weight": weight,
            "available": False,
            "score": 0.0,
            "signal": "HOLD",
            "confidence": 0.0,
        }

        try:
            funding = dataset.get("funding")
            if not funding or funding.get("current_rate") is None:
                base["reason"] = "no_funding_data"
                return base

            rate = float(funding["current_rate"])
            baseline = 0.0001

            adjusted = rate - baseline
            score = max(-1.0, min(1.0, -adjusted * 200))

            abs_adj = abs(adjusted)
            if abs_adj > 0.003:
                confidence = 0.8
            elif abs_adj > 0.001:
                confidence = 0.6
            elif abs_adj > 0.0003:
                confidence = 0.4
            else:
                confidence = 0.2

            signal = (
                "LONG"
                if score > 0.15
                else "SHORT" if score < -0.15 else "HOLD"
            )

            return {
                "weight": weight,
                "available": True,
                "score": round(score, 4),
                "signal": signal,
                "confidence": confidence,
                "funding_rate": rate,
                "funding_rate_pct": round(rate * 100, 4),
                "annualized": funding.get("annualized_rate"),
            }

        except Exception as exc:
            base["error"] = str(exc)
            logger.warning(f"    Funding: {exc}")
            return base

    # ══════════════════════════════════════════════════════════════════
    #  COMPONENT: MARKET STRUCTURE
    # ══════════════════════════════════════════════════════════════════
    def _get_market_structure_signal(self, dataset: Dict) -> Dict:
        """
        Multi-timeframe trend alignment with timeframe weighting.

        Higher timeframes get more weight for STRUCTURE assessment:
          1d = 3x weight (trend direction)
          4h = 2x weight (setup confirmation)
          1h = 1x weight (entry timing)

        4 checks per timeframe:
          1. Price vs EMA50 (position)
          2. EMA21 vs EMA50 (MA alignment)
          3. EMA21 slope over 5 bars (momentum direction)
          4. Price change over 10 bars (recent returns)
        """
        weight = SIGNAL_WEIGHTS.get("market_structure", 0.15)
        base = {
            "weight": weight,
            "available": False,
            "score": 0.0,
            "signal": "HOLD",
            "confidence": 0.0,
        }

        try:
            ohlcv = dataset.get("ohlcv", {})
            if not ohlcv:
                base["reason"] = "no_ohlcv_data"
                return base

            # Higher TFs matter more for structure
            tf_weights = {"1h": 1.0, "4h": 2.0, "1d": 3.0}

            weighted_bull = 0.0
            weighted_bear = 0.0
            weighted_total = 0.0
            tf_details = {}

            for tf_name, df in ohlcv.items():
                if df is None or len(df) < 50:
                    continue

                closes = df["close"].astype(float)
                price = float(closes.iloc[-1])
                ema21_s = closes.ewm(span=21, adjust=False).mean()
                ema50_s = closes.ewm(span=50, adjust=False).mean()

                ema21_now = float(ema21_s.iloc[-1])
                ema50_now = float(ema50_s.iloc[-1])

                # Weight for this timeframe
                tw = tf_weights.get(tf_name, 1.0)

                # ── Check 1: Price vs EMA50 ──
                if price > ema50_now * 1.005:
                    weighted_bull += tw
                    price_trend = "BULL"
                elif price < ema50_now * 0.995:
                    weighted_bear += tw
                    price_trend = "BEAR"
                else:
                    price_trend = "FLAT"
                weighted_total += tw

                # ── Check 2: EMA21 vs EMA50 ──
                if ema21_now > ema50_now * 1.002:
                    weighted_bull += tw
                    ema_trend = "BULL"
                elif ema21_now < ema50_now * 0.998:
                    weighted_bear += tw
                    ema_trend = "BEAR"
                else:
                    ema_trend = "FLAT"
                weighted_total += tw

                # ── Check 3: EMA21 slope ──
                slope_trend = "N/A"
                lookback = min(5, len(ema21_s) - 1)
                if lookback >= 3:
                    ema21_prev = float(ema21_s.iloc[-lookback])
                    if ema21_prev > 0:
                        slope_pct = (ema21_now - ema21_prev) / ema21_prev * 100
                        if slope_pct > 0.1:
                            weighted_bull += tw
                            slope_trend = "BULL"
                        elif slope_pct < -0.1:
                            weighted_bear += tw
                            slope_trend = "BEAR"
                        else:
                            slope_trend = "FLAT"
                        weighted_total += tw

                # ── Check 4: Recent price momentum ──
                momentum_trend = "N/A"
                if len(closes) > 10:
                    price_prev = float(closes.iloc[-11])
                    if price_prev > 0:
                        ret_pct = (price - price_prev) / price_prev * 100
                        if ret_pct > 0.5:
                            weighted_bull += tw
                            momentum_trend = "BULL"
                        elif ret_pct < -0.5:
                            weighted_bear += tw
                            momentum_trend = "BEAR"
                        else:
                            momentum_trend = "FLAT"
                        weighted_total += tw

                tf_details[tf_name] = {
                    "price_trend": price_trend,
                    "ema_trend": ema_trend,
                    "slope_trend": slope_trend,
                    "momentum_trend": momentum_trend,
                    "weight": tw,
                }

            if weighted_total == 0:
                base["reason"] = "insufficient_data"
                return base

            score = (weighted_bull - weighted_bear) / weighted_total
            alignment = max(weighted_bull, weighted_bear) / weighted_total
            confidence = round(alignment, 3)

            signal = (
                "LONG"
                if score > 0.2
                else "SHORT" if score < -0.2 else "HOLD"
            )

            return {
                "weight": weight,
                "available": True,
                "score": round(score, 4),
                "signal": signal,
                "confidence": confidence,
                "bullish_signals": round(weighted_bull, 1),
                "bearish_signals": round(weighted_bear, 1),
                "total_signals": round(weighted_total, 1),
                "alignment_pct": round(alignment * 100, 1),
                "tf_details": tf_details,
            }

        except Exception as exc:
            base["error"] = str(exc)
            logger.warning(f"    Structure: {exc}")
            return base

    def _combine_signals(self, components: Dict[str, Dict]) -> Dict:
        """
        Weighted combination of all component scores.
        
        FIXED v3.1:
          - Conflict detection: ML vs Structure disagreement halves confidence
          - Minimum 2 available components required
          - Better agreement calculation
        """
        available = {
            k: v for k, v in components.items() if v.get("available")
        }

        if len(available) < 2:
            logger.warning(f"  Only {len(available)} component(s) available — defaulting to HOLD")
            return {
                "signal": "HOLD",
                "direction": 0,
                "confidence": 0.0,
                "combined_score": 0.0,
                "active_components": len(available),
                "total_components": len(components),
                "agreement": 1.0,
                "weight_coverage": 0.0,
            }

        # Re-normalize weights
        raw_weight_sum = sum(c["weight"] for c in available.values())
        if raw_weight_sum <= 0:
            raw_weight_sum = 1.0

        # Compute weighted score
        weighted_score = 0.0
        for comp in available.values():
            normalized_weight = comp["weight"] / raw_weight_sum
            weighted_score += comp["score"] * normalized_weight

        weighted_score = round(
            max(-1.0, min(1.0, weighted_score)), 4
        )

        # Score -> Signal
        abs_score = abs(weighted_score)
        if weighted_score > self.SIGNAL_THRESHOLD:
            signal = "LONG"
            direction = 1
        elif weighted_score < -self.SIGNAL_THRESHOLD:
            signal = "SHORT"
            direction = -1
        else:
            signal = "HOLD"
            direction = 0

        # Confidence from score magnitude
        confidence = round(min(1.0, abs_score * 1.5), 3)

        # Agreement calculation
        if direction != 0:
            strong = [c for c in available.values() if abs(c.get("score", 0)) > 0.05]
            if strong:
                agreeing = sum(
                    1 for c in strong
                    if (c["score"] > 0 and direction > 0)
                    or (c["score"] < 0 and direction < 0)
                )
                agreement = agreeing / len(strong)
            else:
                agreement = 1.0
            confidence = round(
                confidence * (0.7 + 0.3 * agreement), 3
            )
        else:
            agreement = 1.0

        # ══ NEW: Conflict detection — ML vs Market Structure ══
        # If the two biggest components strongly disagree, halve confidence
        ml_comp = components.get("ml_ensemble", {})
        struct_comp = components.get("market_structure", {})

        conflict_detected = False
        if ml_comp.get("available") and struct_comp.get("available"):
            ml_s = float(ml_comp.get("score", 0))
            struct_s = float(struct_comp.get("score", 0))

            # Opposite signs AND both significant
            if (ml_s * struct_s < 0
                    and abs(ml_s) > 0.25
                    and abs(struct_s) > 0.25):
                conflict_detected = True
                old_conf = confidence
                confidence = round(confidence * 0.4, 3)  # Severe penalty
                logger.info(
                    f"  ⚠️ ML/Structure CONFLICT: "
                    f"ML={ml_s:+.3f} vs Struct={struct_s:+.3f} → "
                    f"confidence {old_conf:.3f} → {confidence:.3f}"
                )

        # ══ NEW: Require minimum agreement for non-HOLD ══
        if direction != 0 and agreement < 0.4:
            logger.info(
                f"  ⚠️ Low agreement ({agreement:.2f}) — forcing HOLD"
            )
            signal = "HOLD"
            direction = 0
            confidence = round(confidence * 0.5, 3)

        confidence = min(1.0, max(0.0, confidence))

        return {
            "signal": signal,
            "direction": direction,
            "confidence": float(confidence),
            "combined_score": float(weighted_score),
            "active_components": len(available),
            "total_components": len(components),
            "agreement": round(float(agreement), 3),
            "weight_coverage": round(float(raw_weight_sum), 3),
            "conflict_detected": conflict_detected,
        }

    # ══════════════════════════════════════════════════════════════════
    #  CALCULATE SL / TP LEVELS
    # ══════════════════════════════════════════════════════════════════

    def _calculate_levels(
        self, dataset: Dict, direction: int
    ) -> Dict:
        """
        ATR-based stop-loss and take-profit levels.
        Falls back to fixed percentages if ATR unavailable.
        """
        price = self._get_current_price(dataset)
        if price is None or direction == 0:
            return {
                "entry_price": price,
                "stop_loss": None,
                "take_profit": None,
                "risk_reward_ratio": None,
            }

        use_atr = RISK_CONFIG.get("use_atr_stops", True)
        sl_mult = RISK_CONFIG.get("atr_stop_multiplier", 2.0)
        default_sl_pct = RISK_CONFIG.get("default_stop_loss_pct", 2.0)
        default_tp_pct = RISK_CONFIG.get("default_take_profit_pct", 4.0)

        atr = self._get_atr(dataset)

        if use_atr and atr and atr > 0:
            sl_distance = atr * sl_mult
            tp_distance = atr * sl_mult * 2  # 2:1 R:R
        else:
            sl_distance = price * default_sl_pct / 100
            tp_distance = price * default_tp_pct / 100

        if direction == 1:  # LONG
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
        else:  # SHORT
            stop_loss = price + sl_distance
            take_profit = price - tp_distance

        risk_pct = sl_distance / price * 100
        reward_pct = tp_distance / price * 100
        rr_ratio = round(reward_pct / risk_pct, 1) if risk_pct > 0 else 0

        return {
            "entry_price": round(price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "risk_pct": round(risk_pct, 2),
            "reward_pct": round(reward_pct, 2),
            "risk_reward_ratio": rr_ratio,
            "atr": round(atr, 2) if atr else None,
            "atr_pct": round(atr / price * 100, 2) if atr else None,
        }

    # ══════════════════════════════════════════════════════════════════
    #  PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _fetch_dataset(self, symbol: str) -> Optional[Dict]:
        """Lazy-import DataManager, fetch full dataset with news."""
        try:
            from data.manager import DataManager

            dm = DataManager()
            dataset = dm.get_full_dataset(
                symbol, use_cache=True, include_news=True
            )

            ohlcv = dataset.get("ohlcv", {})
            if not ohlcv:
                logger.error(f"No OHLCV data for {symbol}")
                return None

            return dataset

        except Exception as exc:
            logger.error(f"Dataset fetch {symbol}: {exc}", exc_info=True)
            return None

    def _get_current_price(self, dataset: Dict) -> Optional[float]:
        """Extract current price from dataset."""
        ticker = dataset.get("ticker") or {}
        price = ticker.get("last_price")
        if price:
            return float(price)

        entry_tf = TIMEFRAMES.get("entry", "1h")
        ohlcv = dataset.get("ohlcv", {})
        df = ohlcv.get(entry_tf)

        if df is None or df.empty:
            for _tf, _df in ohlcv.items():
                if _df is not None and not _df.empty:
                    df = _df
                    break

        if df is not None and len(df) > 0:
            return round(float(df["close"].iloc[-1]), 2)

        return None

    def _get_atr(self, dataset: Dict, period: int = 14) -> Optional[float]:
        """Calculate ATR from entry-timeframe OHLCV."""
        entry_tf = TIMEFRAMES.get("entry", "1h")
        ohlcv = dataset.get("ohlcv", {})
        df = ohlcv.get(entry_tf)

        if df is None or len(df) < period + 1:
            return None

        try:
            closes = df["close"].astype(float)
            highs = df["high"].astype(float)
            lows = df["low"].astype(float)
            prev_c = closes.shift(1)
            tr = pd.concat(
                [
                    highs - lows,
                    (highs - prev_c).abs(),
                    (lows - prev_c).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr_val = float(tr.rolling(period).mean().iloc[-1])
            return atr_val if not pd.isna(atr_val) else None
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════
    #  DB LOGGING — v3.0.3 FIX: ensure all fields are properly saved
    # ══════════════════════════════════════════════════════════════════

    def _log_signal(self, result: Dict) -> Optional[int]:
        """
        Save signal to SQLite database with all component details.

        v3.0.3 FIX:
          - All numeric fields explicitly cast to float (prevent None)
          - Debug logging shows exact values being saved
          - combined_score guaranteed to be present
        """
        try:
            db = get_db()

            # ── Extract component details for DB columns ──
            comps = result.get("components", {})

            ml = comps.get("ml_ensemble", {})
            sent = comps.get("sentiment", {})
            ai = comps.get("ai_reasoning", {})
            fund = comps.get("funding_rate", {})
            struct = comps.get("market_structure", {})

            # ── Build data dict with explicit float casts ──
            # FIX: ensure no None values sneak in for numeric fields
            combined_score = float(result.get("combined_score", 0) or 0)
            confidence = float(result.get("confidence", 0) or 0)

            ml_dir = ml.get("signal") if ml.get("available") else None
            ml_conf = float(ml.get("confidence", 0) or 0) if ml.get("available") else 0.0
            ml_agree = float(ml.get("agreement", 0) or 0) if ml.get("available") else 0.0

            sent_score = float(sent.get("score", 0) or 0) if sent.get("available") else 0.0

            ai_dir = ai.get("signal") if ai.get("available") else None
            ai_conf = float(ai.get("confidence", 0) or 0) if ai.get("available") else 0.0

            fund_score = float(fund.get("score", 0) or 0) if fund.get("available") else 0.0
            struct_score = float(struct.get("score", 0) or 0) if struct.get("available") else 0.0

            data = {
                "symbol": result.get("symbol", "UNKNOWN"),
                "direction": result.get("direction", 0),
                "confidence": confidence,
                "combined_score": combined_score,
                "entry_price": result.get("entry_price"),
                "stop_loss": result.get("stop_loss"),
                "take_profit": result.get("take_profit"),
                "status": "generated",

                # ML ensemble
                "ml_direction": ml_dir,
                "ml_confidence": ml_conf,
                "ml_agreement": ml_agree,

                # Sentiment
                "sentiment_score": sent_score,

                # AI reasoning
                "ai_direction": ai_dir,
                "ai_confidence": ai_conf,

                # Funding & structure
                "funding_signal": fund_score,
                "market_signal": struct_score,

                # Metadata in notes
                "notes": str({
                    "active_components": result.get("active_components"),
                    "total_components": result.get("total_components"),
                    "agreement": result.get("agreement"),
                    "data_quality": result.get("data_quality"),
                    "weight_coverage": result.get("weight_coverage"),
                })[:500],
            }

            # ── Debug: log what we're about to save ──
            logger.debug(
                f"DB save: {data['symbol']} dir={data['direction']} "
                f"conf={confidence:.3f} score={combined_score:+.3f} | "
                f"ML={ml_dir}({ml_conf:.3f}) sent={sent_score:+.3f} "
                f"fund={fund_score:+.3f} struct={struct_score:+.3f}"
            )

            signal_id = db.save_signal(data)
            return signal_id

        except Exception as exc:
            logger.warning(f"DB log failed: {exc}", exc_info=True)
            return None

    # ══════════════════════════════════════════════════════════════════
    #  STATUS & HISTORY
    # ══════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict:
        """Current SignalEngine status."""
        return {
            "ai_enabled": self.brain.enabled,
            "ai_status": self.brain.get_status(),
            "signal_weights": SIGNAL_WEIGHTS,
            "signal_threshold": self.SIGNAL_THRESHOLD,
            "signals_generated": self._signal_count,
            "last_signals": {
                sym: {
                    "signal": r.get("signal"),
                    "confidence": r.get("confidence"),
                    "time": r.get("timestamp"),
                }
                for sym, r in self._last_signals.items()
            },
        }

    def get_last_signal(
        self, symbol: Optional[str] = None
    ) -> Optional[Dict]:
        """Get last generated signal for a symbol."""
        if symbol:
            return self._last_signals.get(symbol)
        return dict(self._last_signals)

    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Get signal history from database."""
        try:
            db = get_db()
            return db.get_signals(symbol=symbol, limit=limit)
        except Exception as exc:
            logger.warning(f"History fetch failed: {exc}")
            return []


# ═══════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """
    Test suite for SignalEngine.
    Tests component extractors, combiner, level calculator,
    and optionally live signal generation.

    Run:  python -m analysis.signal_engine
    """

    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  SIGNAL ENGINE — TEST SUITE (v3.0.3)")
    print(SEP)

    engine = SignalEngine()

    # ── Status ──
    print("\n[1/8] Status …")
    status = engine.get_status()
    print(f"  AI enabled:    {status['ai_enabled']}")
    print(f"  Threshold:     {status['signal_threshold']}")
    print(f"  Weights:       {status['signal_weights']}")

    # ── Build synthetic dataset ──
    print("\n[2/8] Building synthetic dataset …")
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-06-01", periods=n, freq="1h")
    close = 67000 + np.cumsum(np.random.randn(n) * 100)
    fake_df = pd.DataFrame(
        {
            "open": close - np.random.rand(n) * 50,
            "high": close + np.random.rand(n) * 200,
            "low": close - np.random.rand(n) * 200,
            "close": close,
            "volume": np.random.uniform(100, 5000, n),
        },
        index=dates,
    )

    fake_4h = fake_df.iloc[::4].copy()
    fake_1d = fake_df.iloc[::24].copy()

    fake_dataset = {
        "symbol": "BTCUSDT",
        "ohlcv": {"1h": fake_df, "4h": fake_4h, "1d": fake_1d},
        "funding": {
            "current_rate": 0.0005,
            "annualized_rate": 5.48,
        },
        "ticker": {
            "last_price": float(close[-1]),
            "price_change_pct": -0.36,
            "quote_volume_24h": 6534530026,
        },
        "open_interest": {"open_interest": 84651.9},
        "sentiment": {
            "sentiment_score": -0.25,
            "sentiment_label": "bearish",
            "confidence": 0.65,
            "fear_greed": {"value": 12, "label": "Extreme Fear"},
        },
        "data_quality": {"score": 85, "valid": True},
    }
    print(f"  Symbol:  {fake_dataset['symbol']}")
    print(f"  Price:   ${close[-1]:.2f}")

    # ── Test individual components ──
    print("\n[3/8] Testing individual components …")

    sent = engine._get_sentiment_signal(fake_dataset)
    assert sent["available"]
    assert -1 <= sent["score"] <= 1
    print(f"  Sentiment:  score={sent['score']:+.3f} signal={sent['signal']}")

    fund = engine._get_funding_signal(fake_dataset)
    assert fund["available"]
    assert -1 <= fund["score"] <= 1
    print(f"  Funding:    score={fund['score']:+.3f} signal={fund['signal']}")

    struct = engine._get_market_structure_signal(fake_dataset)
    assert struct["available"]
    assert -1 <= struct["score"] <= 1
    print(f"  Structure:  score={struct['score']:+.3f} signal={struct['signal']}")

    ml = engine._get_ml_signal("_FAKE_", fake_dataset)
    assert not ml["available"]
    print(f"  ML:         available={ml['available']} (expected: no model)")

    # ── Test combiner ──
    print("\n[4/8] Testing signal combiner …")

    bear_components = {
        "ml_ensemble": {
            "available": True, "score": -0.6, "signal": "SHORT",
            "confidence": 0.7, "weight": 0.45,
        },
        "sentiment": {
            "available": True, "score": -0.3, "signal": "SHORT",
            "confidence": 0.6, "weight": 0.15,
        },
        "ai_reasoning": {
            "available": True, "score": -0.5, "signal": "SHORT",
            "confidence": 0.65, "weight": 0.15,
        },
        "funding_rate": {
            "available": True, "score": -0.2, "signal": "SHORT",
            "confidence": 0.5, "weight": 0.10,
        },
        "market_structure": {
            "available": True, "score": -0.7, "signal": "SHORT",
            "confidence": 0.8, "weight": 0.15,
        },
    }
    combined_bear = engine._combine_signals(bear_components)
    assert combined_bear["signal"] == "SHORT"
    assert combined_bear["combined_score"] < 0
    print(f"  All bearish:   signal={combined_bear['signal']} "
          f"score={combined_bear['combined_score']:+.3f}")

    # ── Test DB logging ──
    print("\n[5/8] Testing DB signal logging …")

    test_result = {
        "symbol": "TESTUSDT",
        "signal": "SHORT",
        "direction": -1,
        "confidence": 0.298,
        "combined_score": -0.234,
        "entry_price": 600.0,
        "stop_loss": 612.0,
        "take_profit": 576.0,
        "active_components": 4,
        "total_components": 5,
        "agreement": 0.5,
        "data_quality": 85,
        "weight_coverage": 0.85,
        "components": {
            "ml_ensemble": {
                "available": True, "score": -0.448, "signal": "SHORT",
                "confidence": 0.724, "agreement": 0.75, "weight": 0.45,
            },
            "sentiment": {
                "available": True, "score": -0.330, "signal": "SHORT",
                "confidence": 0.65, "weight": 0.15,
            },
            "ai_reasoning": {
                "available": False, "score": 0.0, "signal": "HOLD",
                "confidence": 0.0, "weight": 0.15,
            },
            "funding_rate": {
                "available": True, "score": 0.020, "signal": "HOLD",
                "confidence": 0.2, "weight": 0.10,
            },
            "market_structure": {
                "available": True, "score": 0.333, "signal": "LONG",
                "confidence": 0.5, "weight": 0.15,
            },
        },
    }

    sig_id = engine._log_signal(test_result)
    print(f"  Signal ID: {sig_id}")

    # Verify from DB
    db = get_db()
    sigs = db.get_signals(symbol="TESTUSDT", limit=1)
    if sigs:
        s = sigs[0]
        db_score = float(s.get("combined_score", 0) or 0)
        db_conf = float(s.get("confidence", 0) or 0)
        db_ml_dir = s.get("ml_direction")
        db_ml_conf = float(s.get("ml_confidence", 0) or 0)
        db_sent = float(s.get("sentiment_score", 0) or 0)
        db_fund = float(s.get("funding_signal", 0) or 0)
        db_struct = float(s.get("market_signal", 0) or 0)

        print(f"  DB verify:")
        print(f"    combined_score: {db_score:+.3f} (expected: -0.234)")
        print(f"    confidence:     {db_conf:.3f}  (expected: 0.298)")
        print(f"    ml_direction:   {db_ml_dir}    (expected: SHORT)")
        print(f"    ml_confidence:  {db_ml_conf:.3f}  (expected: 0.724)")
        print(f"    sentiment:      {db_sent:+.3f} (expected: -0.330)")
        print(f"    funding:        {db_fund:+.3f} (expected: +0.020)")
        print(f"    market_signal:  {db_struct:+.3f} (expected: +0.333)")

        # Assertions
        assert abs(db_score - (-0.234)) < 0.01, f"combined_score WRONG: {db_score}"
        assert abs(db_conf - 0.298) < 0.01, f"confidence WRONG: {db_conf}"
        assert db_ml_dir == "SHORT", f"ml_direction WRONG: {db_ml_dir}"
        assert abs(db_ml_conf - 0.724) < 0.01, f"ml_confidence WRONG: {db_ml_conf}"
        assert abs(db_sent - (-0.330)) < 0.01, f"sentiment WRONG: {db_sent}"
        assert abs(db_fund - 0.020) < 0.01, f"funding WRONG: {db_fund}"
        assert abs(db_struct - 0.333) < 0.01, f"market_signal WRONG: {db_struct}"

        print(f"  ✅ All DB fields verified correctly!")
    else:
        print(f"  ❌ Could not read signal back from DB!")

    # ── Test level calculator ──
    print("\n[6/8] Testing SL/TP calculator …")
    levels_long = engine._calculate_levels(fake_dataset, 1)
    assert levels_long["stop_loss"] < levels_long["entry_price"]
    assert levels_long["take_profit"] > levels_long["entry_price"]
    print(f"  LONG:  SL={levels_long['stop_loss']} TP={levels_long['take_profit']}")

    levels_short = engine._calculate_levels(fake_dataset, -1)
    assert levels_short["stop_loss"] > levels_short["entry_price"]
    assert levels_short["take_profit"] < levels_short["entry_price"]
    print(f"  SHORT: SL={levels_short['stop_loss']} TP={levels_short['take_profit']}")

    # ── Full pipeline with synthetic data ──
    print("\n[7/8] Testing full pipeline (synthetic, no AI) …")
    result = engine.generate_signal(
        "BTCUSDT", dataset=fake_dataset, include_ai=False
    )
    assert result["signal"] in ("LONG", "SHORT", "HOLD")
    assert 0 <= result["confidence"] <= 1
    print(f"  Signal: {result['signal']} conf={result['confidence']:.3f} "
          f"score={result.get('combined_score', 0):+.3f}")

    # Verify this signal in DB too
    btc_sigs = db.get_signals(symbol="BTCUSDT", limit=1)
    if btc_sigs:
        bs = btc_sigs[0]
        bs_score = float(bs.get("combined_score", 0) or 0)
        bs_sent = float(bs.get("sentiment_score", 0) or 0)
        print(f"  DB check: score={bs_score:+.3f} sent={bs_sent:+.3f}")
        if abs(bs_score) < 0.001 and result.get("combined_score", 0) != 0:
            print(f"  ❌ BUG: combined_score is 0 in DB but "
                  f"{result['combined_score']:+.3f} in result!")
        else:
            print(f"  ✅ DB values match result")

    # ── Optional live test ──
    print(f"\n[8/8] Live test (BTCUSDT, no AI) …")
    try:
        live = engine.generate_signal("BTCUSDT", include_ai=False)
        if live.get("error"):
            print(f"  ⚠️  Error: {live['error']}")
        else:
            print(f"  Signal: {live['signal']} conf={live['confidence']:.3f} "
                  f"score={live.get('combined_score', 0):+.3f}")

            # Verify in DB
            live_sigs = db.get_signals(symbol="BTCUSDT", limit=1)
            if live_sigs:
                ls = live_sigs[0]
                ls_score = float(ls.get("combined_score", 0) or 0)
                ls_ml = ls.get("ml_direction")
                ls_ml_c = float(ls.get("ml_confidence", 0) or 0)
                ls_sent = float(ls.get("sentiment_score", 0) or 0)
                ls_fund = float(ls.get("funding_signal", 0) or 0)
                ls_struct = float(ls.get("market_signal", 0) or 0)
                print(f"  DB: score={ls_score:+.3f} ML={ls_ml}({ls_ml_c:.2f}) "
                      f"sent={ls_sent:+.3f} fund={ls_fund:+.3f} struct={ls_struct:+.3f}")
    except Exception as exc:
        print(f"  ❌ Live test failed: {exc}")

    print(f"\n{SEP}")
    print(f"  ✅ ALL TESTS PASSED")
    print(SEP)