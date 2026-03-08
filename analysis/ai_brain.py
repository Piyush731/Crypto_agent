"""
Crypto Futures AI Agent — AI Brain
====================================
HuggingFace-powered AI reasoning layer for crypto futures trading.

Two reasoning methods:
    1. Chain-of-Thought (CoT): 5-step sequential analysis
       Steps 1+2  → Technical + Sentiment   (analyst model)
       Step  3    → Risk assessment          (risk_manager model)
       Steps 4+5  → Synthesis + Verification (strategist model)

    2. Expert Debate: 3 AI "experts" argue perspectives
       Bull case   → analyst model
       Bear case   → strategist model
       Moderator   → risk_manager model verdict

Uses HuggingFace Router API (OpenAI-compatible chat/completions endpoint).
Retry logic with fallback models if primary ones are busy / rate-limited.
Graceful degradation: full → partial → rule-based fallback → disabled.

Some models (DeepSeek R1, Qwen3) return <think>...</think> reasoning blocks
which are stripped before parsing the structured answer.

Data flow:
    signal_engine calls → brain.analyze(dataset, ml_prediction, sentiment_data)
    brain builds context → runs CoT + Debate → combines → returns verdict

Usage:
    brain  = AIBrain()
    result = brain.analyze(dataset, ml_prediction, sentiment_data)
    cot    = brain.chain_of_thought(context_dict)
    debate = brain.expert_debate(context_dict)
"""

import re
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import (
    AI_CONFIG,
    HF_TOKEN,
    TIMEFRAMES,
)
from core.logger import get_logger

logger = get_logger("analysis.ai_brain")


class AIBrain:
    """
    HuggingFace-powered AI reasoning for crypto futures trading.

    Produces: signal (LONG/SHORT/HOLD), confidence (0-1), structured reasoning.
    Used by SignalEngine at 15% weight in final signal composition.
    """

    # New HF Router endpoint — OpenAI-compatible chat/completions
    HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

    # ──────────────────────────────────────────────────────────────────
    #  INIT
    # ──────────────────────────────────────────────────────────────────

    def __init__(self):
        self.enabled = AI_CONFIG.get("enabled", False)
        self.hf_token = HF_TOKEN
        self.models = AI_CONFIG.get("models", {})
        self.fallback_models = AI_CONFIG.get("fallback_models", [])
        self.max_tokens = AI_CONFIG.get("max_new_tokens", 500)
        self.temperature = AI_CONFIG.get("temperature", 0.6)
        self.retry_attempts = AI_CONFIG.get("retry_attempts", 3)
        self.retry_delay = AI_CONFIG.get("retry_delay_seconds", 5)
        self.timeout = AI_CONFIG.get("timeout_seconds", 45)

        # Session counters
        self._models_used: List[str] = []
        self._call_count = 0
        self._error_count = 0

        if self.enabled:
            logger.info(
                "AIBrain initialized | models: "
                + ", ".join(
                    f"{k}={v.split('/')[-1]}" for k, v in self.models.items()
                )
            )
        else:
            logger.warning("AIBrain disabled (no HF_TOKEN)")

    # ══════════════════════════════════════════════════════════════════
    #  MAIN ENTRY — analyze()
    # ══════════════════════════════════════════════════════════════════

    def analyze(
        self,
        dataset: Dict,
        ml_prediction: Optional[Dict] = None,
        sentiment_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Full AI reasoning pipeline: Chain-of-Thought + Expert Debate.

        Parameters
        ----------
        dataset        : dict  from DataManager.get_full_dataset()
        ml_prediction  : dict  from EnsemblePredictor.predict()
        sentiment_data : dict  from SentimentAnalyzer.get_market_sentiment()

        Returns
        -------
        dict  with keys:
            signal          LONG / SHORT / HOLD
            direction       1 / -1 / 0
            confidence      0.0 – 1.0
            reasoning       str summary
            cot_analysis    dict | None
            expert_debate   dict | None
            models_used     List[str]
            method          full / cot_only / debate_only / fallback / disabled
            api_calls       int
            api_errors      int
            analysis_time_s float
            enabled         bool
            agreement       bool   (CoT vs Debate agree?)
        """
        t0 = time.time()

        if not self.enabled:
            return self._disabled_result()

        # Reset session counters
        self._models_used = []
        self._call_count = 0
        self._error_count = 0

        # Build context from all data sources
        context = self._build_context(dataset, ml_prediction, sentiment_data)
        logger.info(
            f"  AI: {context['symbol']} @ ${context.get('price', '?')} | "
            f"RSI={context.get('rsi_14', '?')} | trend={context.get('trend', '?')}"
        )

        # ── Run Chain-of-Thought ──────────────────────────────────────
        cot_result = None
        try:
            cot_result = self.chain_of_thought(context)
        except Exception as exc:
            logger.error(f"CoT failed: {exc}", exc_info=True)
            cot_result = {
                "signal": "HOLD",
                "confidence": 0.0,
                "direction": 0,
                "error": str(exc),
            }

        # ── Run Expert Debate ─────────────────────────────────────────
        debate_result = None
        try:
            debate_result = self.expert_debate(context)
        except Exception as exc:
            logger.error(f"Expert debate failed: {exc}", exc_info=True)
            debate_result = {
                "signal": "HOLD",
                "confidence": 0.0,
                "direction": 0,
                "error": str(exc),
            }

        # ── Combine ──────────────────────────────────────────────────
        combined = self._combine_results(cot_result, debate_result, context)
        combined["analysis_time_s"] = round(time.time() - t0, 2)
        combined["models_used"] = list(set(self._models_used))
        combined["api_calls"] = self._call_count
        combined["api_errors"] = self._error_count
        combined["enabled"] = True

        logger.info(
            f"  AI result: {combined['signal']} | "
            f"conf={combined['confidence']:.2f} | "
            f"method={combined['method']} | "
            f"calls={self._call_count} errors={self._error_count} | "
            f"{combined['analysis_time_s']}s"
        )
        return combined

    # ══════════════════════════════════════════════════════════════════
    #  CHAIN-OF-THOUGHT (5 steps → 3 API calls)
    # ══════════════════════════════════════════════════════════════════

    def chain_of_thought(self, context: Dict) -> Dict:
        """
        5-step Chain-of-Thought reasoning (grouped into 3 API calls).

        Steps:
            1+2  Technical + Sentiment   → analyst model
            3    Risk assessment          → risk_manager model
            4+5  Synthesis + Verification → strategist model

        Returns
        -------
        dict  signal, confidence, direction, steps: {technical, sentiment,
              risk, synthesis, verification}, method
        """
        logger.info("  CoT: starting chain-of-thought …")
        steps: Dict[str, Dict] = {}

        # ── Steps 1+2: Technical + Sentiment (analyst) ────────────────
        analyst_model = self.models.get("analyst", "")
        ts_prompt = self._prompt_tech_sentiment(context)
        ts_response = self._query_model(analyst_model, ts_prompt, "analyst")

        if ts_response:
            parsed = self._parse_tech_sentiment(ts_response)
            steps["technical"] = {
                "trend": parsed.get("trend", "NEUTRAL"),
                "strength": parsed.get("strength", 5),
                "signal": parsed.get("signal", "HOLD"),
                "raw": ts_response[:500],
            }
            steps["sentiment"] = {
                "mood": parsed.get("sentiment_mood", "NEUTRAL"),
                "raw": ts_response[:500],
            }
            logger.info(
                f"    Step 1+2: trend={steps['technical']['trend']} "
                f"mood={steps['sentiment']['mood']}"
            )
        else:
            steps["technical"] = {
                "trend": "NEUTRAL",
                "strength": 5,
                "error": "API failed",
            }
            steps["sentiment"] = {"mood": "NEUTRAL", "error": "API failed"}
            logger.warning("    Step 1+2: analyst API failed")

        # ── Step 3: Risk Assessment (risk_manager) ────────────────────
        risk_model = self.models.get("risk_manager", "")
        risk_prompt = self._prompt_risk(context, steps)
        risk_response = self._query_model(risk_model, risk_prompt, "risk_manager")

        if risk_response:
            parsed = self._parse_risk(risk_response)
            steps["risk"] = {
                "level": parsed.get("risk_level", "MEDIUM"),
                "factors": parsed.get("risk_factors", []),
                "position_adj": parsed.get("position_adjustment", 1.0),
                "raw": risk_response[:500],
            }
            logger.info(f"    Step 3: risk={steps['risk']['level']}")
        else:
            steps["risk"] = {
                "level": "MEDIUM",
                "position_adj": 1.0,
                "error": "API failed",
            }
            logger.warning("    Step 3: risk_manager API failed")

        # ── Steps 4+5: Synthesis + Verification (strategist) ──────────
        strat_model = self.models.get("strategist", "")
        synth_prompt = self._prompt_synthesis(context, steps)
        synth_response = self._query_model(strat_model, synth_prompt, "strategist")

        if synth_response:
            parsed = self._parse_signal(synth_response)
            steps["synthesis"] = {
                "signal": parsed.get("signal", "HOLD"),
                "confidence": parsed.get("confidence", 0.5),
                "reasoning": parsed.get("reasoning", ""),
                "raw": synth_response[:500],
            }
            steps["verification"] = {
                "confirmed": parsed.get("confirmed", True),
                "raw": synth_response[:500],
            }
            logger.info(
                f"    Step 4+5: signal={steps['synthesis']['signal']} "
                f"conf={steps['synthesis']['confidence']:.2f}"
            )
        else:
            steps["synthesis"] = {
                "signal": "HOLD",
                "confidence": 0.5,
                "error": "API failed",
            }
            steps["verification"] = {"confirmed": False, "error": "API failed"}
            logger.warning("    Step 4+5: strategist API failed")

        # ── Aggregate CoT result ──────────────────────────────────────
        cot_signal = steps["synthesis"].get("signal", "HOLD")
        cot_conf = steps["synthesis"].get("confidence", 0.5)

        # Adjust confidence based on risk level
        risk_multiplier = {
            "LOW": 1.1,
            "MEDIUM": 1.0,
            "HIGH": 0.85,
            "EXTREME": 0.7,
        }.get(steps["risk"].get("level", "MEDIUM"), 1.0)
        cot_conf = round(min(1.0, max(0.0, cot_conf * risk_multiplier)), 3)

        result = {
            "signal": cot_signal,
            "confidence": cot_conf,
            "direction": self._signal_to_dir(cot_signal),
            "steps": steps,
            "method": "chain_of_thought",
        }
        logger.info(f"  CoT final: {cot_signal} conf={cot_conf:.2f}")
        return result

    # ══════════════════════════════════════════════════════════════════
    #  EXPERT DEBATE (3 API calls)
    # ══════════════════════════════════════════════════════════════════

    def expert_debate(self, context: Dict) -> Dict:
        """
        3-expert debate: Bull vs Bear → Moderator verdict.

        Returns
        -------
        dict  signal, confidence, direction, bull_case, bear_case, moderator
        """
        logger.info("  Debate: starting expert debate …")

        # ── Bull Case (analyst) ───────────────────────────────────────
        analyst_model = self.models.get("analyst", "")
        bull_prompt = self._prompt_bull(context)
        bull_raw = self._query_model(analyst_model, bull_prompt, "analyst")

        if bull_raw:
            bull = self._parse_debate(bull_raw, "LONG")
            logger.info(
                f"    Bull: conf={bull['confidence']:.2f} "
                f"args={len(bull.get('arguments', []))}"
            )
        else:
            bull = {
                "signal": "LONG",
                "confidence": 0.4,
                "arguments": [],
                "reasoning": "",
                "error": "API failed",
            }
            logger.warning("    Bull: analyst API failed")

        # ── Bear Case (strategist) ───────────────────────────────────
        strat_model = self.models.get("strategist", "")
        bear_prompt = self._prompt_bear(context)
        bear_raw = self._query_model(strat_model, bear_prompt, "strategist")

        if bear_raw:
            bear = self._parse_debate(bear_raw, "SHORT")
            logger.info(
                f"    Bear: conf={bear['confidence']:.2f} "
                f"args={len(bear.get('arguments', []))}"
            )
        else:
            bear = {
                "signal": "SHORT",
                "confidence": 0.4,
                "arguments": [],
                "reasoning": "",
                "error": "API failed",
            }
            logger.warning("    Bear: strategist API failed")

        # ── Moderator Verdict (risk_manager) ──────────────────────────
        risk_model = self.models.get("risk_manager", "")
        mod_prompt = self._prompt_moderator(
            context,
            bull_raw or "Bull analyst unavailable.",
            bear_raw or "Bear analyst unavailable.",
        )
        mod_raw = self._query_model(risk_model, mod_prompt, "risk_manager")

        if mod_raw:
            moderator = self._parse_signal(mod_raw)
            logger.info(
                f"    Moderator: {moderator.get('signal', 'HOLD')} "
                f"conf={moderator.get('confidence', 0.5):.2f}"
            )
        else:
            # Fallback: compare bull vs bear confidence
            if bull["confidence"] > bear["confidence"] + 0.1:
                moderator = {
                    "signal": "LONG",
                    "confidence": round(bull["confidence"] * 0.6, 3),
                }
            elif bear["confidence"] > bull["confidence"] + 0.1:
                moderator = {
                    "signal": "SHORT",
                    "confidence": round(bear["confidence"] * 0.6, 3),
                }
            else:
                moderator = {"signal": "HOLD", "confidence": 0.4}
            moderator["error"] = "API failed — rule fallback"
            logger.warning("    Moderator: risk_manager API failed, used rule fallback")

        debate_signal = moderator.get("signal", "HOLD")
        debate_conf = moderator.get("confidence", 0.5)

        result = {
            "signal": debate_signal,
            "confidence": round(debate_conf, 3),
            "direction": self._signal_to_dir(debate_signal),
            "bull_case": bull,
            "bear_case": bear,
            "moderator": moderator,
            "method": "expert_debate",
        }
        logger.info(f"  Debate final: {debate_signal} conf={debate_conf:.2f}")
        return result

    # ══════════════════════════════════════════════════════════════════
    #  CONTEXT BUILDING
    # ══════════════════════════════════════════════════════════════════

    def _build_context(
        self,
        dataset: Dict,
        ml_prediction: Optional[Dict],
        sentiment_data: Optional[Dict],
    ) -> Dict:
        """Extract key data points from all sources for prompt building."""

        ctx: Dict = {
            "symbol": dataset.get("symbol", "UNKNOWN"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # ── OHLCV — entry timeframe ──────────────────────────────────
        entry_tf = TIMEFRAMES.get("entry", "1h")
        ohlcv = dataset.get("ohlcv", {})
        df = ohlcv.get(entry_tf)

        # Fallback: grab any available timeframe
        if (df is None or (hasattr(df, 'empty') and df.empty)) and ohlcv:
            for _tf, _df in ohlcv.items():
                if _df is not None and not (hasattr(_df, 'empty') and _df.empty):
                    df = _df
                    entry_tf = _tf
                    break

        if df is not None and len(df) >= 2:
            closes = df["close"].astype(float)
            highs = df["high"].astype(float)
            lows = df["low"].astype(float)
            volumes = df["volume"].astype(float)

            ctx["price"] = round(float(closes.iloc[-1]), 2)
            ctx["open"] = round(float(df["open"].iloc[-1]), 2)

            # 24h hi/lo
            tail24 = min(24, len(df))
            ctx["high_24h"] = round(float(highs.tail(tail24).max()), 2)
            ctx["low_24h"] = round(float(lows.tail(tail24).min()), 2)

            # Returns
            ctx["return_1h"] = self._pct(closes, 1)
            ctx["return_24h"] = self._pct(closes, 24)
            ctx["return_7d"] = self._pct(closes, 168)

            # RSI(14)
            ctx["rsi_14"] = self._simple_rsi(closes, 14)

            # EMAs
            ctx["ema_9"] = self._ema_last(closes, 9)
            ctx["ema_21"] = self._ema_last(closes, 21)
            ctx["ema_50"] = self._ema_last(closes, 50)
            ctx["ema_200"] = (
                self._ema_last(closes, 200) if len(closes) >= 200 else None
            )

            # MACD
            if len(closes) >= 26:
                ema12 = closes.ewm(span=12, adjust=False).mean()
                ema26 = closes.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                macd_sig = macd_line.ewm(span=9, adjust=False).mean()
                ctx["macd_hist"] = round(
                    float((macd_line - macd_sig).iloc[-1]), 4
                )
                ctx["macd_status"] = (
                    "BULLISH" if ctx["macd_hist"] > 0 else "BEARISH"
                )
            else:
                ctx["macd_hist"] = 0.0
                ctx["macd_status"] = "NEUTRAL"

            # Bollinger %B
            if len(closes) >= 20:
                sma20 = closes.rolling(20).mean()
                std20 = closes.rolling(20).std()
                bb_upper = sma20 + 2 * std20
                bb_lower = sma20 - 2 * std20
                bb_range = float(bb_upper.iloc[-1] - bb_lower.iloc[-1])
                if bb_range > 0:
                    ctx["bb_pct_b"] = round(
                        float(closes.iloc[-1] - bb_lower.iloc[-1])
                        / bb_range
                        * 100,
                        1,
                    )
                else:
                    ctx["bb_pct_b"] = 50.0
            else:
                ctx["bb_pct_b"] = 50.0

            # ATR(14)
            if len(df) >= 15:
                prev_c = closes.shift(1)
                tr = pd.concat(
                    [
                        highs - lows,
                        (highs - prev_c).abs(),
                        (lows - prev_c).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                atr_val = float(tr.rolling(14).mean().iloc[-1])
                ctx["atr"] = round(atr_val, 2)
                ctx["atr_pct"] = round(atr_val / ctx["price"] * 100, 2)
            else:
                ctx["atr"] = None
                ctx["atr_pct"] = None

            # Volume ratio vs 20-period average
            if len(volumes) >= 20:
                vol_avg = float(volumes.rolling(20).mean().iloc[-1])
                ctx["volume_ratio"] = (
                    round(float(volumes.iloc[-1]) / vol_avg, 2)
                    if vol_avg > 0
                    else 1.0
                )
            else:
                ctx["volume_ratio"] = 1.0

            # Simple trend: price vs EMA-50
            if ctx.get("ema_50"):
                ratio = ctx["price"] / ctx["ema_50"]
                if ratio > 1.01:
                    ctx["trend"] = "BULLISH"
                elif ratio < 0.99:
                    ctx["trend"] = "BEARISH"
                else:
                    ctx["trend"] = "NEUTRAL"
            else:
                ctx["trend"] = "NEUTRAL"

        else:
            ctx["price"] = None
            ctx["trend"] = "UNKNOWN"

        # ── Higher-timeframe trends ──────────────────────────────────
        for role, tf_interval in TIMEFRAMES.items():
            if tf_interval == entry_tf:
                continue
            htf_df = ohlcv.get(tf_interval)
            if htf_df is not None and len(htf_df) >= 50:
                htf_c = htf_df["close"].astype(float)
                htf_ema50 = float(
                    htf_c.ewm(span=50, adjust=False).mean().iloc[-1]
                )
                r = (
                    float(htf_c.iloc[-1]) / htf_ema50
                    if htf_ema50
                    else 1.0
                )
                if r > 1.01:
                    ctx[f"tf_{tf_interval}_trend"] = "BULLISH"
                elif r < 0.99:
                    ctx[f"tf_{tf_interval}_trend"] = "BEARISH"
                else:
                    ctx[f"tf_{tf_interval}_trend"] = "NEUTRAL"

        # ── Funding rate ─────────────────────────────────────────────
        funding = dataset.get("funding") or {}
        ctx["funding_rate"] = funding.get("current_rate", 0.0)
        ctx["funding_ann"] = funding.get("annualized_rate", 0.0)

        # ── Ticker ───────────────────────────────────────────────────
        ticker = dataset.get("ticker") or {}
        ctx["price_change_24h"] = ticker.get("price_change_pct", 0.0)
        ctx["volume_24h_usd"] = ticker.get("quote_volume_24h", 0)

        # ── ML Prediction ────────────────────────────────────────────
        if ml_prediction:
            ctx["ml_signal"] = ml_prediction.get("signal", "HOLD")
            ctx["ml_confidence"] = ml_prediction.get("confidence", 0.5)
            ctx["ml_agreement"] = ml_prediction.get("agreement", 0.5)
            ctx["ml_prob_up"] = ml_prediction.get("probability_up", 0.5)
        else:
            ctx["ml_signal"] = "N/A"
            ctx["ml_confidence"] = 0.0

        # ── Sentiment ────────────────────────────────────────────────
        if sentiment_data:
            ctx["sent_score"] = sentiment_data.get("sentiment_score", 0.0)
            ctx["sent_label"] = sentiment_data.get(
                "sentiment_label", "neutral"
            )
            fg = sentiment_data.get("fear_greed") or {}
            ctx["fg_value"] = fg.get("value", 50)
            ctx["fg_label"] = fg.get("label", "Neutral")
        else:
            ctx["sent_score"] = 0.0
            ctx["sent_label"] = "N/A"
            ctx["fg_value"] = 50
            ctx["fg_label"] = "Neutral"

        return ctx

    # ── small helpers for context ────────────────────────────────────

    @staticmethod
    def _pct(series: pd.Series, periods: int) -> float:
        """Percentage return over `periods` bars, safely."""
        if len(series) <= periods:
            return 0.0
        prev = float(series.iloc[-(periods + 1)])
        if prev == 0:
            return 0.0
        return round(float((series.iloc[-1] / prev - 1) * 100), 2)

    @staticmethod
    def _ema_last(series: pd.Series, span: int) -> Optional[float]:
        if len(series) < span:
            return None
        val = series.ewm(span=span, adjust=False).mean().iloc[-1]
        return round(float(val), 2) if not pd.isna(val) else None

    @staticmethod
    def _simple_rsi(series: pd.Series, period: int = 14) -> float:
        try:
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_g = gain.rolling(period).mean()
            avg_l = loss.rolling(period).mean()
            rs = avg_g / avg_l.replace(0, 1e-10)
            rsi = 100 - 100 / (1 + rs)
            v = rsi.iloc[-1]
            return round(float(v), 1) if not pd.isna(v) else 50.0
        except Exception:
            return 50.0

    # ══════════════════════════════════════════════════════════════════
    #  PROMPT BUILDERS
    # ══════════════════════════════════════════════════════════════════

    def _prompt_tech_sentiment(self, c: Dict) -> str:
        """CoT steps 1+2 — technical + sentiment analysis."""
        return (
            f"You are a crypto futures analyst. Analyze {c['symbol']} and respond "
            f"in the EXACT format shown below. Do NOT add extra commentary.\n\n"
            f"MARKET DATA:\n"
            f"- Price: ${c.get('price', '?')} | 24h Change: {c.get('price_change_24h', 0):.2f}%\n"
            f"- RSI(14): {c.get('rsi_14', '?')} | MACD: {c.get('macd_status', '?')} "
            f"(hist: {c.get('macd_hist', 0):.4f})\n"
            f"- EMA21: ${c.get('ema_21', '?')} | EMA50: ${c.get('ema_50', '?')}\n"
            f"- BB%B: {c.get('bb_pct_b', 50):.0f}% | VolRatio: {c.get('volume_ratio', 1):.1f}x\n"
            f"- Trend: {c.get('trend', '?')} | ATR%: {c.get('atr_pct', '?')}%\n"
            f"- Funding: {c.get('funding_rate', 0):.4f}%\n\n"
            f"SENTIMENT:\n"
            f"- Score: {c.get('sent_score', 0):.2f} ({c.get('sent_label', '?')})\n"
            f"- Fear&Greed: {c.get('fg_value', 50)} ({c.get('fg_label', '?')})\n"
            f"- ML model: {c.get('ml_signal', '?')} ({c.get('ml_confidence', 0):.0%})\n\n"
            f"RESPOND EXACTLY:\n"
            f"TECHNICAL_TREND: [BULLISH/BEARISH/NEUTRAL]\n"
            f"TECHNICAL_STRENGTH: [1-10]\n"
            f"SENTIMENT_MOOD: [BULLISH/BEARISH/NEUTRAL]\n"
            f"SIGNAL: [LONG/SHORT/HOLD]\n"
            f"CONFIDENCE: [0-100]\n"
            f"REASONING: [2-3 sentences]"
        )

    def _prompt_risk(self, c: Dict, steps: Dict) -> str:
        """CoT step 3 — risk assessment."""
        tech_trend = steps.get("technical", {}).get("trend", "NEUTRAL")
        sent_mood = steps.get("sentiment", {}).get("mood", "NEUTRAL")
        return (
            f"You are a crypto risk manager. Assess risk for {c['symbol']}. "
            f"Respond in the EXACT format below. Do NOT add extra commentary.\n\n"
            f"- Price: ${c.get('price', '?')} | ATR%: {c.get('atr_pct', '?')}%\n"
            f"- 24h Range: ${c.get('low_24h', '?')} – ${c.get('high_24h', '?')}\n"
            f"- Funding: {c.get('funding_rate', 0):.4f}% | VolRatio: {c.get('volume_ratio', 1):.1f}x\n"
            f"- Fear&Greed: {c.get('fg_value', 50)} ({c.get('fg_label', '?')})\n"
            f"- Technical: {tech_trend} | Sentiment: {sent_mood}\n\n"
            f"RESPOND EXACTLY:\n"
            f"RISK_LEVEL: [LOW/MEDIUM/HIGH/EXTREME]\n"
            f"POSITION_ADJUSTMENT: [0.5/0.75/1.0/1.25/1.5]\n"
            f"KEY_RISKS: [2-3 bullet points]\n"
            f"REASONING: [2-3 sentences]"
        )

    def _prompt_synthesis(self, c: Dict, steps: Dict) -> str:
        """CoT steps 4+5 — synthesis + verification."""
        tech = steps.get("technical", {})
        risk = steps.get("risk", {})
        return (
            f"You are a senior crypto strategist. Make a FINAL trading decision "
            f"for {c['symbol']} and verify your logic. "
            f"Respond in the EXACT format below. Do NOT add extra commentary.\n\n"
            f"ANALYSIS:\n"
            f"- Technical: {tech.get('trend', '?')} (strength {tech.get('strength', 5)}/10)\n"
            f"- Sentiment: {steps.get('sentiment', {}).get('mood', '?')}\n"
            f"- Risk: {risk.get('level', '?')}\n"
            f"- ML: {c.get('ml_signal', '?')} ({c.get('ml_confidence', 0):.0%})\n"
            f"- Price: ${c.get('price', '?')} | RSI: {c.get('rsi_14', '?')} | "
            f"Funding: {c.get('funding_rate', 0):.4f}%\n\n"
            f"Be conservative: only LONG/SHORT if evidence is strong.\n\n"
            f"RESPOND EXACTLY:\n"
            f"SIGNAL: [LONG/SHORT/HOLD]\n"
            f"CONFIDENCE: [0-100]\n"
            f"REASONING: [3-4 sentences with synthesis AND verification]"
        )

    def _prompt_bull(self, c: Dict) -> str:
        """Expert debate — bull case."""
        return (
            f"You are a BULLISH crypto analyst. Make the STRONGEST bullish case "
            f"for {c['symbol']} at ${c.get('price', '?')}. "
            f"Respond in the EXACT format below.\n\n"
            f"Data: RSI={c.get('rsi_14', '?')}, MACD={c.get('macd_status', '?')}, "
            f"Trend={c.get('trend', '?')}, Funding={c.get('funding_rate', 0):.4f}%, "
            f"Fear&Greed={c.get('fg_value', 50)}, 24hChg={c.get('price_change_24h', 0):.2f}%\n\n"
            f"RESPOND EXACTLY:\n"
            f"BULL_CONFIDENCE: [0-100]\n"
            f"UPSIDE_TARGET: [percentage]\n"
            f"KEY_ARGUMENTS: [3 bullet points]\n"
            f"REASONING: [2-3 sentences why LONG]"
        )

    def _prompt_bear(self, c: Dict) -> str:
        """Expert debate — bear case."""
        return (
            f"You are a BEARISH crypto analyst. Make the STRONGEST bearish case "
            f"for {c['symbol']} at ${c.get('price', '?')}. "
            f"Respond in the EXACT format below.\n\n"
            f"Data: RSI={c.get('rsi_14', '?')}, MACD={c.get('macd_status', '?')}, "
            f"Trend={c.get('trend', '?')}, Funding={c.get('funding_rate', 0):.4f}%, "
            f"Fear&Greed={c.get('fg_value', 50)}, 24hChg={c.get('price_change_24h', 0):.2f}%\n\n"
            f"RESPOND EXACTLY:\n"
            f"BEAR_CONFIDENCE: [0-100]\n"
            f"DOWNSIDE_TARGET: [percentage]\n"
            f"KEY_ARGUMENTS: [3 bullet points]\n"
            f"REASONING: [2-3 sentences why SHORT]"
        )

    def _prompt_moderator(self, c: Dict, bull_text: str, bear_text: str) -> str:
        """Expert debate — moderator verdict."""
        return (
            f"You are a NEUTRAL crypto moderator. Two experts debated "
            f"{c['symbol']} at ${c.get('price', '?')}. "
            f"Respond in the EXACT format below.\n\n"
            f"BULL:\n{bull_text[:350]}\n\n"
            f"BEAR:\n{bear_text[:350]}\n\n"
            f"ML Signal: {c.get('ml_signal', '?')}, Fear&Greed: {c.get('fg_value', 50)}\n\n"
            f"RESPOND EXACTLY:\n"
            f"SIGNAL: [LONG/SHORT/HOLD]\n"
            f"CONFIDENCE: [0-100]\n"
            f"WINNER: [BULL/BEAR/NEITHER]\n"
            f"REASONING: [2-3 sentences objective verdict]"
        )

    # ══════════════════════════════════════════════════════════════════
    #  HUGGINGFACE ROUTER API — chat/completions with retry + fallback
    # ══════════════════════════════════════════════════════════════════

    def _query_model(
        self,
        model_id: str,
        prompt: str,
        role: str = "",
    ) -> Optional[str]:
        """
        Query HF Router API.  Tries primary model, then each fallback.
        Returns cleaned text (think blocks stripped) or None on total failure.
        """
        if not model_id:
            logger.warning(f"No model ID for role '{role}'")
            return None

        result = self._try_model(model_id, prompt)
        if result:
            self._models_used.append(model_id)
            return result

        # Fallback chain
        for fb in self.fallback_models:
            logger.info(f"    fallback → {fb.split('/')[-1]}")
            result = self._try_model(fb, prompt)
            if result:
                self._models_used.append(fb)
                return result

        logger.warning(f"  All models failed for role='{role}'")
        return None

    def _try_model(self, model_id: str, prompt: str) -> Optional[str]:
        """
        Try a single HF model with retries using chat/completions endpoint.
        Returns cleaned text or None.
        """
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": max(0.01, self.temperature),
            "stream": False,
        }

        for attempt in range(1, self.retry_attempts + 1):
            try:
                self._call_count += 1
                resp = requests.post(
                    self.HF_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    text = self._extract_text(data)
                    if text:
                        # Strip <think>...</think> blocks
                        text = self._strip_think_blocks(text)
                        if text:
                            return text
                    logger.debug(
                        f"    empty body from {model_id.split('/')[-1]}"
                    )

                elif resp.status_code == 503:
                    # Model loading
                    try:
                        body = resp.json()
                        wait = min(
                            body.get("estimated_time", self.retry_delay), 30
                        )
                    except Exception:
                        wait = self.retry_delay
                    logger.info(
                        f"    {model_id.split('/')[-1]} loading ~{wait:.0f}s "
                        f"(attempt {attempt})"
                    )
                    time.sleep(wait)

                elif resp.status_code == 429:
                    logger.warning(
                        f"    rate-limited {model_id.split('/')[-1]} "
                        f"(attempt {attempt})"
                    )
                    time.sleep(self.retry_delay * attempt)

                elif resp.status_code == 422:
                    logger.warning(
                        f"    invalid input {model_id.split('/')[-1]}: "
                        f"{resp.text[:200]}"
                    )
                    self._error_count += 1
                    return None  # don't retry on input errors

                elif resp.status_code in (401, 403):
                    logger.warning(
                        f"    auth error {model_id.split('/')[-1]} "
                        f"HTTP {resp.status_code}"
                    )
                    self._error_count += 1
                    return None  # don't retry auth errors

                else:
                    logger.warning(
                        f"    {model_id.split('/')[-1]} HTTP {resp.status_code} "
                        f"(attempt {attempt}): {resp.text[:150]}"
                    )
                    self._error_count += 1
                    if attempt < self.retry_attempts:
                        time.sleep(self.retry_delay)

            except requests.exceptions.Timeout:
                logger.warning(
                    f"    timeout {model_id.split('/')[-1]} (attempt {attempt})"
                )
                self._error_count += 1
            except requests.exceptions.ConnectionError:
                logger.warning(
                    f"    conn error {model_id.split('/')[-1]} "
                    f"(attempt {attempt})"
                )
                self._error_count += 1
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay)
            except Exception as exc:
                logger.warning(
                    f"    {model_id.split('/')[-1]} error: {exc} "
                    f"(attempt {attempt})"
                )
                self._error_count += 1

        return None

    @staticmethod
    def _extract_text(data: dict) -> Optional[str]:
        """
        Extract content from OpenAI-compatible chat/completions response.

        Expected format:
            {"choices": [{"message": {"content": "..."}}]}
        """
        try:
            choices = data.get("choices")
            if choices and isinstance(choices, list):
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                if content:
                    return content.strip()

            # Fallback: some providers use slightly different keys
            if "generated_text" in data:
                return data["generated_text"].strip() or None

            if isinstance(data, list) and data:
                item = data[0]
                if isinstance(item, dict):
                    for key in ("generated_text", "text", "content"):
                        if key in item:
                            return item[key].strip() or None
        except Exception:
            pass

        return None

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        """
        Strip <think>...</think> reasoning blocks from model output.
        DeepSeek R1, Qwen3, QwQ put chain-of-thought inside these tags.
        We want only the final structured answer after the tags.
        """
        # Remove all <think>...</think> blocks (greedy, handles multiline)
        cleaned = re.sub(
            r"<think>.*?</think>", "", text, flags=re.DOTALL
        )
        # Also handle unclosed <think> tags (model got cut off mid-thinking)
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
        cleaned = cleaned.strip()
        if cleaned:
            return cleaned
        # Everything was inside think blocks — no actual answer produced
        # Strip the tags but keep inner content as last resort
        fallback = re.sub(r"</?think>", "", text).strip()
        return fallback if fallback else ""

    # ══════════════════════════════════════════════════════════════════
    #  RESPONSE PARSERS
    # ══════════════════════════════════════════════════════════════════

    def _parse_tech_sentiment(self, text: str) -> Dict:
        """Parse CoT steps 1+2 (tech + sentiment) response."""
        r: Dict = {}

        m = re.search(
            r"TECHNICAL_TREND:\s*(BULLISH|BEARISH|NEUTRAL)", text, re.I
        )
        r["trend"] = m.group(1).upper() if m else self._infer_direction(text)

        m = re.search(r"TECHNICAL_STRENGTH:\s*(\d+)", text)
        r["strength"] = min(10, max(1, int(m.group(1)))) if m else 5

        m = re.search(
            r"SENTIMENT_MOOD:\s*(BULLISH|BEARISH|NEUTRAL)", text, re.I
        )
        r["sentiment_mood"] = (
            m.group(1).upper() if m else self._infer_direction(text)
        )

        m = re.search(r"SIGNAL:\s*(LONG|SHORT|HOLD|BUY|SELL)", text, re.I)
        if m:
            sig = m.group(1).upper()
            r["signal"] = {"BUY": "LONG", "SELL": "SHORT"}.get(sig, sig)
        else:
            r["signal"] = "HOLD"

        r["confidence"] = self._extract_confidence(text)

        m = re.search(r"REASONING:\s*(.+?)(?:\n\n|$)", text, re.S)
        r["reasoning"] = m.group(1).strip()[:500] if m else ""

        return r

    def _parse_risk(self, text: str) -> Dict:
        """Parse CoT step 3 (risk) response."""
        r: Dict = {}

        m = re.search(
            r"RISK_LEVEL:\s*(LOW|MEDIUM|HIGH|EXTREME)", text, re.I
        )
        r["risk_level"] = m.group(1).upper() if m else "MEDIUM"

        m = re.search(r"POSITION_ADJUSTMENT:\s*([\d.]+)", text)
        if m:
            r["position_adjustment"] = max(0.5, min(1.5, float(m.group(1))))
        else:
            adj_map = {
                "LOW": 1.25,
                "MEDIUM": 1.0,
                "HIGH": 0.75,
                "EXTREME": 0.5,
            }
            r["position_adjustment"] = adj_map.get(r["risk_level"], 1.0)

        m = re.search(
            r"KEY_RISKS:\s*(.+?)(?:REASONING|$)", text, re.S | re.I
        )
        if m:
            lines = [
                ln.strip().lstrip("-•*0123456789.) ").strip()
                for ln in m.group(1).strip().split("\n")
                if ln.strip()
            ]
            r["risk_factors"] = lines[:5]
        else:
            r["risk_factors"] = []

        return r

    def _parse_signal(self, text: str) -> Dict:
        """Parse synthesis / moderator response → signal + confidence."""
        r: Dict = {}

        m = re.search(r"SIGNAL:\s*(LONG|SHORT|HOLD|BUY|SELL)", text, re.I)
        if m:
            sig = m.group(1).upper()
            r["signal"] = {"BUY": "LONG", "SELL": "SHORT"}.get(sig, sig)
        else:
            r["signal"] = self._infer_signal(text)

        r["confidence"] = self._extract_confidence(text)

        m = re.search(r"REASONING:\s*(.+?)(?:\n\n|$)", text, re.S)
        r["reasoning"] = m.group(1).strip()[:500] if m else text[:300]

        m = re.search(r"WINNER:\s*(BULL|BEAR|NEITHER)", text, re.I)
        if m:
            r["winner"] = m.group(1).upper()

        r["confirmed"] = "REVISE" not in text.upper()

        return r

    def _parse_debate(self, text: str, default_signal: str) -> Dict:
        """Parse bull/bear debate response."""
        r: Dict = {"signal": default_signal}

        for pat in [
            r"BULL_CONFIDENCE:\s*(\d+)",
            r"BEAR_CONFIDENCE:\s*(\d+)",
            r"CONFIDENCE:\s*(\d+)",
        ]:
            m = re.search(pat, text, re.I)
            if m:
                r["confidence"] = (
                    min(100, max(0, int(m.group(1)))) / 100.0
                )
                break
        else:
            r["confidence"] = 0.5

        for pat in [
            r"UPSIDE_TARGET:\s*([\d.]+)",
            r"DOWNSIDE_TARGET:\s*([\d.]+)",
        ]:
            m = re.search(pat, text, re.I)
            if m:
                r["target_pct"] = float(m.group(1))
                break

        m = re.search(
            r"KEY_ARGUMENTS:\s*(.+?)(?:REASONING|$)", text, re.S | re.I
        )
        if m:
            r["arguments"] = [
                ln.strip().lstrip("-•*0123456789.) ").strip()
                for ln in m.group(1).strip().split("\n")
                if ln.strip()
            ][:5]
        else:
            r["arguments"] = []

        m = re.search(r"REASONING:\s*(.+?)(?:\n\n|$)", text, re.S)
        r["reasoning"] = m.group(1).strip()[:500] if m else text[:300]

        r["raw"] = text[:500]
        return r

    # ── Shared parsing helpers ───────────────────────────────────────

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence value from text, return 0-1 float."""
        # Try explicit CONFIDENCE: N pattern
        m = re.search(
            r"CONFIDENCE:\s*(\d+(?:\.\d+)?)\s*%?", text, re.I
        )
        if m:
            val = float(m.group(1))
            if val > 1.0:
                val /= 100.0
            return round(min(1.0, max(0.0, val)), 3)

        # Try "X% confident" or "X/100"
        m = re.search(r"(\d{1,3})\s*%\s*confiden", text, re.I)
        if m:
            return round(int(m.group(1)) / 100.0, 3)

        m = re.search(r"(\d{1,3})\s*/\s*100", text, re.I)
        if m:
            return round(int(m.group(1)) / 100.0, 3)

        # Keyword fallback
        low = text.lower()
        if any(
            w in low
            for w in [
                "very high confidence",
                "extremely confident",
                "strong conviction",
            ]
        ):
            return 0.85
        if any(
            w in low for w in ["high confidence", "confident", "strong"]
        ):
            return 0.7
        if any(w in low for w in ["moderate", "medium", "somewhat"]):
            return 0.55
        if any(
            w in low
            for w in ["low confidence", "uncertain", "unclear", "mixed"]
        ):
            return 0.4

        return 0.5

    def _infer_direction(self, text: str) -> str:
        """Infer BULLISH/BEARISH/NEUTRAL from free-form text."""
        low = text.lower()
        bull_words = [
            "bullish", "uptrend", "buy", "long", "oversold",
            "accumulation", "breakout", "support", "recovery",
            "upside", "rally",
        ]
        bear_words = [
            "bearish", "downtrend", "sell", "short", "overbought",
            "distribution", "breakdown", "resistance", "decline",
            "downside", "crash", "dump",
        ]

        bull_count = sum(1 for w in bull_words if w in low)
        bear_count = sum(1 for w in bear_words if w in low)

        if bull_count > bear_count + 1:
            return "BULLISH"
        if bear_count > bull_count + 1:
            return "BEARISH"
        return "NEUTRAL"

    def _infer_signal(self, text: str) -> str:
        """Infer LONG/SHORT/HOLD from free-form text when parsing fails."""
        low = text.lower()

        long_words = [
            "long", "buy", "bullish signal", "go long", "enter long",
        ]
        short_words = [
            "short", "sell", "bearish signal", "go short", "enter short",
        ]
        hold_words = [
            "hold", "wait", "neutral", "no trade", "stay flat", "sideline",
        ]

        long_score = sum(1 for w in long_words if w in low)
        short_score = sum(1 for w in short_words if w in low)
        hold_score = sum(1 for w in hold_words if w in low)

        top = max(long_score, short_score, hold_score)
        if top == 0:
            return "HOLD"
        if long_score == top and long_score > short_score:
            return "LONG"
        if short_score == top and short_score > long_score:
            return "SHORT"
        return "HOLD"

    # ══════════════════════════════════════════════════════════════════
    #  COMBINE COT + DEBATE → final AI verdict
    # ══════════════════════════════════════════════════════════════════

    def _combine_results(
        self,
        cot: Optional[Dict],
        debate: Optional[Dict],
        context: Dict,
    ) -> Dict:
        """
        Combine Chain-of-Thought and Expert Debate into one AI verdict.

        Weighting:  CoT = 60%, Debate = 40%  (CoT is more structured)
        Agreement bonus: +0.05 confidence if both agree on direction.
        Disagreement penalty: -0.05 confidence if they conflict.
        """
        cot_sig = (cot or {}).get("signal", "HOLD")
        cot_conf = (cot or {}).get("confidence", 0.0)
        cot_ok = cot is not None and "error" not in cot

        debate_sig = (debate or {}).get("signal", "HOLD")
        debate_conf = (debate or {}).get("confidence", 0.0)
        debate_ok = debate is not None and "error" not in debate

        # Determine method used
        if cot_ok and debate_ok:
            method = "full"
        elif cot_ok:
            method = "cot_only"
        elif debate_ok:
            method = "debate_only"
        else:
            method = "fallback"

        # ── Both available: weighted combination ──────────────────────
        if cot_ok and debate_ok:
            cot_weight = 0.6
            debate_weight = 0.4

            cot_dir = self._signal_to_dir(cot_sig)
            debate_dir = self._signal_to_dir(debate_sig)

            # Weighted confidence score
            combined_conf = (
                cot_conf * cot_weight + debate_conf * debate_weight
            )

            # Agreement check
            agree = cot_dir == debate_dir

            if agree and cot_dir != 0:
                combined_conf += 0.05
                final_signal = cot_sig
            elif agree and cot_dir == 0:
                final_signal = "HOLD"
            else:
                combined_conf -= 0.05
                if cot_conf >= debate_conf:
                    final_signal = cot_sig
                else:
                    final_signal = debate_sig

                if combined_conf < 0.45:
                    final_signal = "HOLD"

            combined_conf = round(min(1.0, max(0.0, combined_conf)), 3)

        # ── Only CoT available ────────────────────────────────────────
        elif cot_ok:
            final_signal = cot_sig
            combined_conf = round(cot_conf * 0.85, 3)
            agree = True

        # ── Only Debate available ─────────────────────────────────────
        elif debate_ok:
            final_signal = debate_sig
            combined_conf = round(debate_conf * 0.85, 3)
            agree = True

        # ── Neither available: rule-based fallback ────────────────────
        else:
            fallback = self._rule_based_fallback(context)
            final_signal = fallback["signal"]
            combined_conf = fallback["confidence"]
            agree = True

        # Build reasoning summary
        reasoning_parts = []
        if cot_ok:
            synth = (cot or {}).get("steps", {}).get("synthesis", {})
            if synth.get("reasoning"):
                reasoning_parts.append(f"CoT: {synth['reasoning'][:200]}")
        if debate_ok:
            mod = (debate or {}).get("moderator", {})
            if mod.get("reasoning"):
                reasoning_parts.append(
                    f"Debate: {mod['reasoning'][:200]}"
                )
        if not reasoning_parts:
            reasoning_parts.append(
                f"Rule-based: {context.get('trend', 'NEUTRAL')} trend, "
                f"RSI={context.get('rsi_14', 50)}"
            )

        return {
            "signal": final_signal,
            "direction": self._signal_to_dir(final_signal),
            "confidence": combined_conf,
            "reasoning": " | ".join(reasoning_parts),
            "cot_analysis": cot,
            "expert_debate": debate,
            "method": method,
            "agreement": agree,
        }

    # ══════════════════════════════════════════════════════════════════
    #  RULE-BASED FALLBACK (when all API calls fail)
    # ══════════════════════════════════════════════════════════════════

    def _rule_based_fallback(self, context: Dict) -> Dict:
        """
        Simple rule-based signal when HF API is completely unavailable.
        Uses RSI, trend, funding rate, sentiment — no ML model needed.
        """
        logger.info("  Using rule-based fallback (all AI models failed)")

        score = 0.0
        reasons = []

        # RSI
        rsi = context.get("rsi_14", 50)
        if rsi < 30:
            score += 0.3
            reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi > 70:
            score -= 0.3
            reasons.append(f"RSI overbought ({rsi:.0f})")
        elif rsi < 45:
            score += 0.1
        elif rsi > 55:
            score -= 0.1

        # Trend
        trend = context.get("trend", "NEUTRAL")
        if trend == "BULLISH":
            score += 0.2
            reasons.append("Bullish trend")
        elif trend == "BEARISH":
            score -= 0.2
            reasons.append("Bearish trend")

        # MACD
        macd_hist = context.get("macd_hist", 0)
        if macd_hist > 0:
            score += 0.1
        elif macd_hist < 0:
            score -= 0.1

        # Funding rate (contrarian)
        fr = context.get("funding_rate", 0)
        if fr > 0.01:
            score -= 0.1
            reasons.append("High funding (crowded long)")
        elif fr < -0.01:
            score += 0.1
            reasons.append("Negative funding (crowded short)")

        # Sentiment
        sent = context.get("sent_score", 0)
        score += sent * 0.15

        # Fear & Greed (contrarian at extremes)
        fg = context.get("fg_value", 50)
        if fg < 20:
            score += 0.1
            reasons.append(f"Extreme Fear ({fg})")
        elif fg > 80:
            score -= 0.1
            reasons.append(f"Extreme Greed ({fg})")

        # Convert score to signal
        if score > 0.2:
            signal = "LONG"
        elif score < -0.2:
            signal = "SHORT"
        else:
            signal = "HOLD"

        confidence = round(min(0.6, 0.3 + abs(score) * 0.5), 3)

        return {
            "signal": signal,
            "confidence": confidence,
            "direction": self._signal_to_dir(signal),
            "score": round(score, 3),
            "reasons": reasons,
            "method": "rule_fallback",
        }

    # ══════════════════════════════════════════════════════════════════
    #  DISABLED RESULT
    # ══════════════════════════════════════════════════════════════════

    def _disabled_result(self) -> Dict:
        """Return when AI is disabled (no HF_TOKEN)."""
        return {
            "signal": "HOLD",
            "direction": 0,
            "confidence": 0.0,
            "reasoning": "AI reasoning disabled (no HF_TOKEN)",
            "cot_analysis": None,
            "expert_debate": None,
            "models_used": [],
            "method": "disabled",
            "api_calls": 0,
            "api_errors": 0,
            "analysis_time_s": 0.0,
            "enabled": False,
            "agreement": True,
        }

    # ══════════════════════════════════════════════════════════════════
    #  UTILITY
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _signal_to_dir(signal: str) -> int:
        """LONG → 1, SHORT → -1, HOLD → 0."""
        return {"LONG": 1, "SHORT": -1}.get(signal, 0)

    def get_status(self) -> Dict:
        """Current AIBrain status."""
        return {
            "enabled": self.enabled,
            "has_token": bool(self.hf_token),
            "models": {
                k: v.split("/")[-1] for k, v in self.models.items()
            },
            "fallbacks": [
                m.split("/")[-1] for m in self.fallback_models
            ],
            "session_calls": self._call_count,
            "session_errors": self._error_count,
            "models_used": list(set(self._models_used)),
        }

    def get_info(self) -> Dict:
        """Configuration info."""
        return {
            "enabled": self.enabled,
            "models": self.models,
            "fallback_models": self.fallback_models,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "retry_attempts": self.retry_attempts,
            "timeout_seconds": self.timeout,
        }


# ═══════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """
    Test suite for AIBrain.
    Tests context building, parsing, think-block stripping,
    rule fallback, and optionally live HF Router API.

    Run:  python -m analysis.ai_brain
    """

    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  AI BRAIN — TEST SUITE")
    print(SEP)

    brain = AIBrain()

    # ── Status ────────────────────────────────────────────────────────
    print("\n[1/9] Status …")
    status = brain.get_status()
    print(f"  Enabled:    {status['enabled']}")
    print(f"  Has token:  {status['has_token']}")
    print(f"  Models:     {status['models']}")
    print(f"  Fallbacks:  {status['fallbacks']}")
    print(f"  API URL:    {brain.HF_API_URL}")

    # ── Build fake dataset for testing ────────────────────────────────
    print("\n[2/9] Building synthetic context …")
    import pandas as pd
    import numpy as np

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

    fake_dataset = {
        "symbol": "BTCUSDT",
        "ohlcv": {
            "1h": fake_df,
            "4h": fake_df.iloc[::4].copy(),
            "1d": fake_df.iloc[::24].copy(),
        },
        "funding": {"current_rate": 0.0012, "annualized_rate": 1.05},
        "ticker": {
            "price_change_pct": -0.36,
            "quote_volume_24h": 6534530026,
        },
        "open_interest": {"open_interest": 84651.9},
    }

    fake_ml = {
        "signal": "SHORT",
        "confidence": 0.68,
        "agreement": 0.75,
        "probability_up": 0.32,
    }

    fake_sentiment = {
        "sentiment_score": -0.15,
        "sentiment_label": "bearish",
        "fear_greed": {"value": 12, "label": "Extreme Fear"},
    }

    ctx = brain._build_context(fake_dataset, fake_ml, fake_sentiment)
    print(f"  Symbol:     {ctx['symbol']}")
    print(f"  Price:      ${ctx.get('price', '?')}")
    print(f"  RSI:        {ctx.get('rsi_14', '?')}")
    print(f"  Trend:      {ctx.get('trend', '?')}")
    print(
        f"  MACD:       {ctx.get('macd_status', '?')} "
        f"({ctx.get('macd_hist', 0):.4f})"
    )
    print(f"  BB%B:       {ctx.get('bb_pct_b', '?')}")
    print(f"  Funding:    {ctx.get('funding_rate', '?')}")
    print(
        f"  ML signal:  {ctx.get('ml_signal', '?')} "
        f"({ctx.get('ml_confidence', 0):.0%})"
    )
    print(
        f"  Sentiment:  {ctx.get('sent_score', 0):.2f} "
        f"({ctx.get('sent_label', '?')})"
    )
    print(
        f"  F&G:        {ctx.get('fg_value', '?')} "
        f"({ctx.get('fg_label', '?')})"
    )
    assert ctx["symbol"] == "BTCUSDT"
    assert ctx["price"] is not None
    assert ctx["rsi_14"] is not None

    # ── Test <think> block stripping ──────────────────────────────────
    print("\n[3/9] Testing <think> block stripping …")

    # Normal case with think block
    t1 = (
        "<think>\nLet me analyze this...\nRSI is oversold.\n</think>\n\n"
        "SIGNAL: SHORT\nCONFIDENCE: 72\nREASONING: Bearish."
    )
    c1 = brain._strip_think_blocks(t1)
    assert "<think>" not in c1
    assert "SIGNAL: SHORT" in c1
    print(f"  Normal:     '{c1[:60]}…' ✅")

    # Multiple think blocks
    t2 = (
        "<think>first thought</think>some text"
        "<think>second thought</think>SIGNAL: LONG"
    )
    c2 = brain._strip_think_blocks(t2)
    assert "<think>" not in c2
    assert "SIGNAL: LONG" in c2
    print(f"  Multiple:   '{c2[:60]}' ✅")

    # Unclosed think block (model cut off mid-thinking)
    t3 = "<think>I'm still thinking about this and the output was cut"
    c3 = brain._strip_think_blocks(t3)
    assert "<think>" not in c3
    print(f"  Unclosed:   '{c3[:60]}' (fallback to original) ✅")

    # No think blocks
    t4 = "SIGNAL: HOLD\nCONFIDENCE: 50\nREASONING: Flat market."
    c4 = brain._strip_think_blocks(t4)
    assert c4 == t4
    print(f"  Clean:      unchanged ✅")

    # ── Test parsers ──────────────────────────────────────────────────
    print("\n[4/9] Testing response parsers …")

    sample_ts = (
        "TECHNICAL_TREND: BEARISH\n"
        "TECHNICAL_STRENGTH: 7\n"
        "SENTIMENT_MOOD: BEARISH\n"
        "SIGNAL: SHORT\n"
        "CONFIDENCE: 72\n"
        "REASONING: Price is below EMA50, MACD bearish crossover. "
        "Fear & Greed at extreme fear supports caution."
    )
    parsed_ts = brain._parse_tech_sentiment(sample_ts)
    assert parsed_ts["trend"] == "BEARISH"
    assert parsed_ts["strength"] == 7
    assert parsed_ts["sentiment_mood"] == "BEARISH"
    assert parsed_ts["signal"] == "SHORT"
    assert 0.7 <= parsed_ts["confidence"] <= 0.75
    print(
        f"  Tech/Sent:  trend={parsed_ts['trend']} "
        f"strength={parsed_ts['strength']} "
        f"mood={parsed_ts['sentiment_mood']} "
        f"signal={parsed_ts['signal']} "
        f"conf={parsed_ts['confidence']}"
    )

    sample_risk = (
        "RISK_LEVEL: HIGH\n"
        "POSITION_ADJUSTMENT: 0.75\n"
        "KEY_RISKS:\n"
        "- High volatility environment\n"
        "- Extreme fear may cause whipsaws\n"
        "- Funding rate elevated\n"
        "REASONING: Market is highly volatile with extreme fear."
    )
    parsed_risk = brain._parse_risk(sample_risk)
    assert parsed_risk["risk_level"] == "HIGH"
    assert parsed_risk["position_adjustment"] == 0.75
    assert len(parsed_risk["risk_factors"]) >= 2
    print(
        f"  Risk:       level={parsed_risk['risk_level']} "
        f"adj={parsed_risk['position_adjustment']} "
        f"factors={len(parsed_risk['risk_factors'])}"
    )

    sample_signal = (
        "SIGNAL: SHORT\n"
        "CONFIDENCE: 65\n"
        "WINNER: BEAR\n"
        "REASONING: Both technical and sentiment lean bearish."
    )
    parsed_sig = brain._parse_signal(sample_signal)
    assert parsed_sig["signal"] == "SHORT"
    assert 0.6 <= parsed_sig["confidence"] <= 0.7
    assert parsed_sig.get("winner") == "BEAR"
    print(
        f"  Signal:     signal={parsed_sig['signal']} "
        f"conf={parsed_sig['confidence']} "
        f"winner={parsed_sig.get('winner')}"
    )

    sample_bull = (
        "BULL_CONFIDENCE: 55\n"
        "UPSIDE_TARGET: 5.2\n"
        "KEY_ARGUMENTS:\n"
        "- RSI oversold bounce likely\n"
        "- Extreme fear historically precedes rallies\n"
        "- Support level holding\n"
        "REASONING: Contrarian opportunity with extreme fear reading."
    )
    parsed_bull = brain._parse_debate(sample_bull, "LONG")
    assert parsed_bull["signal"] == "LONG"
    assert 0.5 <= parsed_bull["confidence"] <= 0.6
    assert len(parsed_bull["arguments"]) >= 2
    print(
        f"  Debate:     signal={parsed_bull['signal']} "
        f"conf={parsed_bull['confidence']} "
        f"args={len(parsed_bull['arguments'])}"
    )

    parsed_bad = brain._parse_signal(
        "This is total garbage with no format."
    )
    assert parsed_bad["signal"] in ("LONG", "SHORT", "HOLD")
    assert 0.0 <= parsed_bad["confidence"] <= 1.0
    print(
        f"  Malformed:  signal={parsed_bad['signal']} "
        f"conf={parsed_bad['confidence']} ✅"
    )

    # ── Test _extract_text with new format ────────────────────────────
    print("\n[5/9] Testing chat/completions response extraction …")

    # Standard OpenAI format
    resp1 = {
        "choices": [
            {"message": {"content": "SIGNAL: LONG\nCONFIDENCE: 75"}}
        ]
    }
    assert brain._extract_text(resp1) == "SIGNAL: LONG\nCONFIDENCE: 75"
    print("  OpenAI format:  ✅")

    # Empty content
    resp2 = {"choices": [{"message": {"content": ""}}]}
    assert brain._extract_text(resp2) is None
    print("  Empty content:  ✅")

    # Legacy format fallback
    resp3 = {"generated_text": "SIGNAL: HOLD"}
    assert brain._extract_text(resp3) == "SIGNAL: HOLD"
    print("  Legacy format:  ✅")

    # Garbage
    resp4 = {"random": "stuff"}
    assert brain._extract_text(resp4) is None
    print("  Garbage input:  ✅")

    # ── Test confidence extractor ─────────────────────────────────────
    print("\n[6/9] Testing confidence extraction …")
    assert brain._extract_confidence("CONFIDENCE: 72") == 0.72
    assert brain._extract_confidence("CONFIDENCE: 0.85") == 0.85
    assert brain._extract_confidence("I am 65% confident") == 0.65
    assert brain._extract_confidence("Score: 80/100") == 0.8
    assert (
        brain._extract_confidence("very high confidence in this") == 0.85
    )
    assert brain._extract_confidence("no numbers here") == 0.5
    print("  All confidence extraction tests passed ✅")

    # ── Test direction inference ──────────────────────────────────────
    print("\n[7/9] Testing direction inference …")
    assert (
        brain._infer_direction("Strong bullish breakout uptrend")
        == "BULLISH"
    )
    assert (
        brain._infer_direction("Bearish breakdown sell decline dump")
        == "BEARISH"
    )
    assert (
        brain._infer_direction("The market is flat today") == "NEUTRAL"
    )
    assert brain._infer_signal("go long buy bullish") == "LONG"
    assert brain._infer_signal("go short sell bearish signal") == "SHORT"
    assert brain._infer_signal("wait hold neutral stay flat") == "HOLD"
    print("  All direction inference tests passed ✅")

    # ── Test rule-based fallback ──────────────────────────────────────
    print("\n[8/9] Testing rule-based fallback …")
    fallback = brain._rule_based_fallback(ctx)
    assert fallback["signal"] in ("LONG", "SHORT", "HOLD")
    assert 0.0 <= fallback["confidence"] <= 1.0
    assert fallback["method"] == "rule_fallback"
    print(
        f"  Fallback:   signal={fallback['signal']} "
        f"conf={fallback['confidence']:.3f} "
        f"score={fallback['score']:.3f}"
    )
    print(f"  Reasons:    {fallback['reasons']}")

    # ── Test combine results ──────────────────────────────────────────
    print("\n[9/9] Testing result combination …")

    cot_r = {
        "signal": "SHORT",
        "confidence": 0.7,
        "steps": {"synthesis": {"reasoning": "Bearish momentum"}},
    }
    debate_r = {
        "signal": "SHORT",
        "confidence": 0.65,
        "moderator": {"reasoning": "Bear wins debate"},
    }
    combined = brain._combine_results(cot_r, debate_r, ctx)
    assert combined["signal"] == "SHORT"
    assert combined["agreement"] is True
    assert combined["method"] == "full"
    print(
        f"  Agree:      signal={combined['signal']} "
        f"conf={combined['confidence']:.3f} "
        f"agree={combined['agreement']}"
    )

    cot_r2 = {
        "signal": "LONG",
        "confidence": 0.6,
        "steps": {"synthesis": {"reasoning": "Contrarian buy"}},
    }
    debate_r2 = {
        "signal": "SHORT",
        "confidence": 0.7,
        "moderator": {"reasoning": "Trend is down"},
    }
    combined2 = brain._combine_results(cot_r2, debate_r2, ctx)
    assert combined2["agreement"] is False
    assert combined2["method"] == "full"
    print(
        f"  Disagree:   signal={combined2['signal']} "
        f"conf={combined2['confidence']:.3f} "
        f"agree={combined2['agreement']}"
    )

    combined3 = brain._combine_results(cot_r, None, ctx)
    assert combined3["method"] == "cot_only"
    print(
        f"  CoT only:   signal={combined3['signal']} "
        f"conf={combined3['confidence']:.3f} "
        f"method={combined3['method']}"
    )

    combined4 = brain._combine_results(
        {"signal": "HOLD", "confidence": 0.0, "error": "fail"},
        {"signal": "HOLD", "confidence": 0.0, "error": "fail"},
        ctx,
    )
    assert combined4["method"] == "fallback"
    print(
        f"  Fallback:   signal={combined4['signal']} "
        f"conf={combined4['confidence']:.3f} "
        f"method={combined4['method']}"
    )

    # ╔════════════════════════════════════════════════════════════════╗
    # ║  LIVE HF ROUTER API TEST — only runs if HF_TOKEN is set      ║
    # ╚════════════════════════════════════════════════════════════════╝
    if brain.enabled:
        print(f"\n{'─' * 40}")
        print("  LIVE TEST: Querying HuggingFace Router API …")
        print(f"{'─' * 40}")

        result = brain.analyze(fake_dataset, fake_ml, fake_sentiment)
        print(f"  Signal:     {result['signal']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Direction:  {result['direction']}")
        print(f"  Method:     {result['method']}")
        print(f"  Agreement:  {result['agreement']}")
        print(f"  API calls:  {result['api_calls']}")
        print(f"  API errors: {result['api_errors']}")
        print(f"  Models:     {result['models_used']}")
        print(f"  Time:       {result['analysis_time_s']}s")
        print(f"  Reasoning:  {result['reasoning'][:200]}")

        if result.get("cot_analysis"):
            cot = result["cot_analysis"]
            steps = cot.get("steps", {})
            print(f"\n  CoT steps:")
            print(
                f"    Tech:     "
                f"{steps.get('technical', {}).get('trend', '?')}"
            )
            print(
                f"    Sent:     "
                f"{steps.get('sentiment', {}).get('mood', '?')}"
            )
            print(
                f"    Risk:     "
                f"{steps.get('risk', {}).get('level', '?')}"
            )
            print(
                f"    Synth:    "
                f"{steps.get('synthesis', {}).get('signal', '?')} "
                f"({steps.get('synthesis', {}).get('confidence', 0):.2f})"
            )

        if result.get("expert_debate"):
            deb = result["expert_debate"]
            print(f"\n  Debate:")
            print(
                f"    Bull:     "
                f"conf={deb.get('bull_case', {}).get('confidence', 0):.2f}"
            )
            print(
                f"    Bear:     "
                f"conf={deb.get('bear_case', {}).get('confidence', 0):.2f}"
            )
            print(
                f"    Verdict:  {deb.get('signal', '?')} "
                f"({deb.get('confidence', 0):.2f})"
            )
    else:
        print(f"\n  ⚠️  HF_TOKEN not set — skipping live API test")
        print(f"  Set HF_TOKEN in .env to test live HuggingFace queries")

    print(f"\n{SEP}")
    print(f"  ✅ ALL TESTS PASSED")
    print(SEP)