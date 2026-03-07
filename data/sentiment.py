"""
data/sentiment.py - FinBERT sentiment analysis on crypto news headlines

Priority chain:
  1. Local FinBERT (ProsusAI/finbert) via transformers
  2. HuggingFace Inference API (same model, remote)
  3. Keyword-based fallback (always works, lower confidence)

Outputs sentiment on -1 (bearish) to +1 (bullish) scale.
Time-decay weights newer headlines more heavily.
Combines news sentiment (70%) + Fear & Greed index (30%).
"""

import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from core.logger import get_logger
import config

logger = get_logger("sentiment")


class SentimentAnalyzer:
    """
    Multi-method sentiment analyzer for crypto news headlines.

    Methods (priority order):
        1. Local FinBERT via ``transformers`` pipeline
        2. HuggingFace Inference API  (needs HF_TOKEN)
        3. Weighted keyword matching  (always available)
    """

    # ── crypto-specific keyword dictionaries ────────────────────────
    _BULLISH: Dict[str, float] = {
        # strong (+2)
        "surge": 2, "soar": 2, "skyrocket": 2, "breakout": 2,
        "all-time high": 2, "ath": 2, "moon": 2, "parabolic": 2,
        "massive rally": 2, "explosion": 2,
        # moderate (+1.5)
        "rally": 1.5, "pump": 1.5, "bullish": 1.5, "institutional": 1.5,
        "adoption": 1.5, "approval": 1.5, "etf approved": 1.5,
        "upgrade": 1.5, "accumulate": 1.5, "buy signal": 1.5,
        "golden cross": 1.5,
        # mild (+1)
        "rise": 1, "gain": 1, "green": 1, "recover": 1, "bounce": 1,
        "support": 1, "uptrend": 1, "growth": 1, "positive": 1,
        "optimistic": 1, "partnership": 1, "launch": 1, "milestone": 1,
        "record": 1, "demand": 1, "inflow": 1, "buy": 1, "higher": 1,
    }

    _BEARISH: Dict[str, float] = {
        # strong (-2)
        "crash": 2, "collapse": 2, "plunge": 2, "liquidation": 2,
        "capitulation": 2, "death cross": 2, "hack": 2, "exploit": 2,
        "scam": 2, "fraud": 2, "ban": 2, "bankrupt": 2, "insolvent": 2,
        # moderate (-1.5)
        "dump": 1.5, "bearish": 1.5, "sell-off": 1.5, "selloff": 1.5,
        "sec lawsuit": 1.5, "regulation": 1.5, "crackdown": 1.5,
        "fud": 1.5, "bubble": 1.5, "ponzi": 1.5, "correction": 1.5,
        "breakdown": 1.5,
        # mild (-1)
        "drop": 1, "fall": 1, "decline": 1, "sell": 1, "red": 1,
        "fear": 1, "risk": 1, "warning": 1, "concern": 1,
        "resistance": 1, "downtrend": 1, "weak": 1, "outflow": 1,
        "loss": 1, "negative": 1, "investigate": 1, "subpoena": 1,
        "lower": 1,
    }

    # ── constructor ─────────────────────────────────────────────────
    def __init__(self):
        cfg = getattr(config, "SENTIMENT_CONFIG", {})
        self.model_name: str = cfg.get("model_name", "ProsusAI/finbert")
        self.max_headlines: int = cfg.get("max_headlines", 50)
        self.cache_ttl: int = cfg.get("cache_ttl", 1800)
        self.decay_hours: float = cfg.get("decay_hours", 48)
        self.min_confidence: float = cfg.get("min_confidence", 0.3)

        self.hf_token: Optional[str] = getattr(config, "HF_TOKEN", None)
        self.cache_dir = getattr(config, "CACHE_DIR", None)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # model state (lazy)
        self._pipeline = None
        self._model_loaded: bool = False
        self._model_tried: bool = False
        self._active_method: str = "not_initialized"

        # in-memory result cache  {md5 → result}
        self._cache: Dict[str, Dict] = {}
        self._cache_ts: Dict[str, float] = {}

        logger.info("SentimentAnalyzer initialized")

    # ── model loading ───────────────────────────────────────────────
    def _load_model(self) -> bool:
        """Try to load local FinBERT.  Called once (lazy)."""
        if self._model_tried:
            return self._model_loaded
        self._model_tried = True

        try:
            from transformers import pipeline as hf_pipeline  # type: ignore

            logger.info(f"Loading FinBERT model: {self.model_name} …")
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1,
                truncation=True,
                max_length=512,
            )
            # quick smoke test
            if self._pipeline("Bitcoin price rises"):
                self._model_loaded = True
                self._active_method = "finbert_local"
                logger.info("✅ FinBERT loaded (local)")
                return True
        except ImportError:
            logger.warning("transformers not installed – will use fallback")
        except Exception as exc:
            logger.warning(f"FinBERT local load failed: {exc}")

        self._model_loaded = False
        return False

    # ── cache helpers ───────────────────────────────────────────────
    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def _get_cached(self, text: str) -> Optional[Dict]:
        key = self._cache_key(text)
        if key in self._cache:
            if time.time() - self._cache_ts.get(key, 0) < self.cache_ttl:
                return self._cache[key]
            del self._cache[key]
            del self._cache_ts[key]
        return None

    def _put_cache(self, text: str, result: Dict) -> None:
        key = self._cache_key(text)
        self._cache[key] = result
        self._cache_ts[key] = time.time()

    # ── analysis back-ends ──────────────────────────────────────────
    def _finbert_local(self, text: str) -> Optional[Dict]:
        """Analyse *text* with the local FinBERT pipeline."""
        if not self._model_loaded and not self._load_model():
            return None
        try:
            items = self._pipeline(text, top_k=None)  # type: ignore[misc]
            if not items:
                return None
            scores = {it["label"].lower(): float(it["score"]) for it in items}
            return self._scores_to_result(scores, "finbert_local")
        except Exception as exc:
            logger.error(f"FinBERT local error: {exc}")
            return None

    def _finbert_api(self, text: str) -> Optional[Dict]:
        """Analyse *text* via the HuggingFace Inference API."""
        if not self.hf_token:
            return None
        try:
            import requests

            url = f"https://api-inference.huggingface.co/models/{self.model_name}"
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            resp = requests.post(url, headers=headers, json={"inputs": text}, timeout=30)

            # model cold-start
            if resp.status_code == 503:
                wait = min(resp.json().get("estimated_time", 20), 30)
                logger.info(f"HF API model loading – waiting {wait:.0f}s …")
                time.sleep(wait)
                resp = requests.post(url, headers=headers, json={"inputs": text}, timeout=30)

            if resp.status_code != 200:
                logger.warning(f"HF API {resp.status_code}: {resp.text[:200]}")
                return None

            data = resp.json()
            if not data:
                return None
            items = data[0] if isinstance(data[0], list) else data
            scores = {it["label"].lower(): float(it["score"]) for it in items}
            return self._scores_to_result(scores, "finbert_api")

        except Exception as exc:
            logger.warning(f"HF API error: {exc}")
            return None

    def _keyword_fallback(self, text: str) -> Dict:
        """Weighted keyword sentiment (always succeeds)."""
        low = text.lower()
        bull = sum(w for kw, w in self._BULLISH.items() if kw in low)
        bear = sum(w for kw, w in self._BEARISH.items() if kw in low)
        total = bull + bear

        if total == 0:
            return {
                "score": 0.0,
                "confidence": 0.1,
                "label": "neutral",
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "method": "keyword",
            }

        raw = (bull - bear) / total
        conf = min(total / 6.0, 0.85)
        label = "neutral" if abs(raw) < 0.15 else ("bullish" if raw > 0 else "bearish")

        return {
            "score": round(raw, 4),
            "confidence": round(conf, 4),
            "label": label,
            "positive": round(bull / total, 4),
            "negative": round(bear / total, 4),
            "neutral": round(1.0 - conf, 4),
            "method": "keyword",
        }

    # ── shared score → result mapping ───────────────────────────────
    @staticmethod
    def _scores_to_result(scores: Dict[str, float], method: str) -> Dict:
        pos = scores.get("positive", 0.0)
        neg = scores.get("negative", 0.0)
        neu = scores.get("neutral", 0.0)
        raw = pos - neg
        conf = max(pos, neg)
        label = "neutral" if abs(raw) < 0.1 else ("bullish" if raw > 0 else "bearish")
        return {
            "score": round(raw, 4),
            "confidence": round(conf, 4),
            "label": label,
            "positive": round(pos, 4),
            "negative": round(neg, 4),
            "neutral": round(neu, 4),
            "method": method,
        }

    # ── time decay ──────────────────────────────────────────────────
    def _time_decay(self, published: str) -> float:
        """Exponential decay weight (1.0 = now, →0.1 for old)."""
        if not published:
            return 0.5
        try:
            pub = None
            for fmt in (
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S",
                "%a, %d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    pub = datetime.strptime(published, fmt)
                    break
                except ValueError:
                    continue
            if pub is None:
                return 0.5
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            hours = max((datetime.now(timezone.utc) - pub).total_seconds() / 3600.0, 0.0)
            return max(0.1, min(1.0, float(np.exp(-hours / self.decay_hours))))
        except Exception:
            return 0.5

    # ── empty sentinel ──────────────────────────────────────────────
    @staticmethod
    def _empty_headlines_result() -> Dict:
        return {
            "overall_score": 0.0,
            "overall_label": "neutral",
            "overall_confidence": 0.0,
            "headline_count": 0,
            "analyzed_count": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "avg_score": 0.0,
            "weighted_score": 0.0,
            "method": "none",
            "details": [],
        }

    # ═══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════════════

    def analyze_headline(self, headline: str) -> Dict:
        """
        Analyse a single headline.

        Returns
        -------
        dict
            score  : float  –1 … +1
            confidence : float  0 … 1
            label  : "bullish" | "bearish" | "neutral"
            method : str
            positive / negative / neutral : float
            text   : str (truncated)
        """
        if not headline or not headline.strip():
            return {
                "score": 0.0, "confidence": 0.0, "label": "neutral",
                "method": "none", "text": "",
                "positive": 0.0, "negative": 0.0, "neutral": 1.0,
            }

        headline = headline.strip()

        cached = self._get_cached(headline)
        if cached is not None:
            return cached

        result = (
            self._finbert_local(headline)
            or self._finbert_api(headline)
            or self._keyword_fallback(headline)
        )

        result["text"] = headline[:120]
        self._put_cache(headline, result)
        return result

    def analyze_headlines(self, headlines: List[Dict]) -> Dict:
        """
        Analyse a batch of headline dicts and aggregate.

        Parameters
        ----------
        headlines : list[dict]
            Each dict must have ``title``; optionally ``published``, ``source``.

        Returns
        -------
        dict
            overall_score, overall_label, overall_confidence,
            headline_count, analyzed_count,
            bullish_count, bearish_count, neutral_count,
            avg_score, weighted_score, method, details
        """
        if not headlines:
            return self._empty_headlines_result()

        headlines = headlines[: self.max_headlines]

        details: List[Dict] = []
        scores: List[float] = []
        weights: List[float] = []
        methods_seen: set = set()
        bull = bear = neut = 0

        for item in headlines:
            title = item.get("title", "")
            if not title:
                continue

            res = self.analyze_headline(title)

            decay = self._time_decay(item.get("published", ""))
            res["source"] = item.get("source", "unknown")
            res["published"] = item.get("published", "")
            res["decay_weight"] = round(decay, 4)

            details.append(res)
            scores.append(res["score"])
            weights.append(decay)
            methods_seen.add(res.get("method", "unknown"))

            if res["label"] == "bullish":
                bull += 1
            elif res["label"] == "bearish":
                bear += 1
            else:
                neut += 1

        n = len(scores)
        if n == 0:
            return self._empty_headlines_result()

        avg_score = float(np.mean(scores))

        total_w = sum(weights)
        weighted_score = (
            sum(s * w for s, w in zip(scores, weights)) / total_w
            if total_w > 0
            else avg_score
        )

        score_std = float(np.std(scores)) if n > 1 else 0.5
        agreement = 1.0 - min(score_std, 1.0)
        count_factor = min(n / 10.0, 1.0)
        overall_conf = agreement * 0.6 + count_factor * 0.4

        if abs(weighted_score) < 0.1:
            overall_label = "neutral"
        elif weighted_score > 0:
            overall_label = "bullish"
        else:
            overall_label = "bearish"

        primary = (
            "finbert_local" if "finbert_local" in methods_seen
            else "finbert_api" if "finbert_api" in methods_seen
            else "keyword"
        )

        return {
            "overall_score": round(weighted_score, 4),
            "overall_label": overall_label,
            "overall_confidence": round(overall_conf, 4),
            "headline_count": len(headlines),
            "analyzed_count": n,
            "bullish_count": bull,
            "bearish_count": bear,
            "neutral_count": neut,
            "avg_score": round(avg_score, 4),
            "weighted_score": round(weighted_score, 4),
            "method": primary,
            "details": details,
        }

    def get_market_sentiment(self, news_data: Dict) -> Dict:
        """
        Comprehensive market sentiment from news + Fear & Greed.

        Parameters
        ----------
        news_data : dict
            Output of ``NewsData.get_all_news()`` with keys
            *headlines*, *fear_greed*, *source_counts*, *total_headlines*.

        Returns
        -------
        dict
            sentiment_score  (-1…+1, final combined),
            sentiment_label, confidence,
            news_sentiment   (full ``analyze_headlines`` output),
            fear_greed, fear_greed_normalized (-1…+1),
            fear_greed_label, combined_method, analyzed_at
        """
        headlines = news_data.get("headlines", [])
        fear_greed = news_data.get("fear_greed", None)

        # ── headlines ───────────────────────────────────────────────
        news_sent = self.analyze_headlines(headlines)

        # ── fear & greed  (0-100 → -1…+1) ──────────────────────────
        fg_norm = 0.0
        fg_label = "neutral"
        fg_ok = False
        if fear_greed and fear_greed.get("value") is not None:
            try:
                fg_val = int(fear_greed["value"])
                fg_norm = (fg_val - 50) / 50.0
                fg_label = fear_greed.get("label", "neutral")
                fg_ok = True
            except (ValueError, TypeError):
                pass

        # ── combine  (news 70 % + FG 30 %) ─────────────────────────
        has_news = news_sent["analyzed_count"] > 0
        if has_news and fg_ok:
            combined = news_sent["weighted_score"] * 0.7 + fg_norm * 0.3
            conf = news_sent["overall_confidence"] * 0.7 + 0.8 * 0.3
            method = f"{news_sent['method']}+fear_greed"
        elif has_news:
            combined = news_sent["weighted_score"]
            conf = news_sent["overall_confidence"]
            method = news_sent["method"]
        elif fg_ok:
            combined = fg_norm
            conf = 0.5
            method = "fear_greed_only"
        else:
            combined = 0.0
            conf = 0.0
            method = "none"

        if abs(combined) < 0.1:
            label = "neutral"
        elif combined > 0:
            label = "bullish"
        else:
            label = "bearish"

        result = {
            "sentiment_score": round(combined, 4),
            "sentiment_label": label,
            "confidence": round(conf, 4),
            "news_sentiment": news_sent,
            "fear_greed": fear_greed,
            "fear_greed_normalized": round(fg_norm, 4),
            "fear_greed_label": fg_label,
            "combined_method": method,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            f"Market sentiment: {label} "
            f"(score={combined:+.3f}, conf={conf:.3f}, "
            f"headlines={news_sent['analyzed_count']}, method={method})"
        )
        return result

    # ── housekeeping ────────────────────────────────────────────────
    def get_cache_info(self) -> Dict:
        """Return cache / model status."""
        return {
            "cached_headlines": len(self._cache),
            "model_loaded": self._model_loaded,
            "active_method": self._active_method,
            "model_name": self.model_name,
        }

    def clear_cache(self) -> None:
        """Flush the in-memory headline cache."""
        self._cache.clear()
        self._cache_ts.clear()
        logger.info("Sentiment cache cleared")


# ═══════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("SENTIMENT ANALYZER TEST")
    print("=" * 60)

    sa = SentimentAnalyzer()

    # ── 1. single headlines ─────────────────────────────────────────
    print("\n📰  Test 1: Individual Headlines")
    print("-" * 50)

    samples = [
        "Bitcoin surges past $100,000 as institutional adoption accelerates",
        "Crypto market crashes amid regulatory crackdown fears",
        "Ethereum completes major network upgrade successfully",
        "SEC files lawsuit against major cryptocurrency exchange",
        "Bitcoin trading volume remains steady as market consolidates",
        "Major bank announces Bitcoin custody service launch",
        "Crypto exchange hacked, millions stolen from users",
        "Bitcoin ETF sees record inflows for third consecutive week",
    ]

    for h in samples:
        r = sa.analyze_headline(h)
        icon = "🟢" if r["label"] == "bullish" else "🔴" if r["label"] == "bearish" else "⚪"
        print(f"  {icon} [{r['score']:+.3f}] ({r['method']:14s}) {h[:65]}")

    # ── 2. batch aggregation ────────────────────────────────────────
    print(f"\n📊  Test 2: Batch Aggregation")
    print("-" * 50)

    mock_hl = [{"title": h, "source": "test", "published": "", "url": ""} for h in samples]
    batch = sa.analyze_headlines(mock_hl)

    print(f"  Overall score : {batch['overall_score']:+.4f}")
    print(f"  Label         : {batch['overall_label']}")
    print(f"  Confidence    : {batch['overall_confidence']:.4f}")
    print(f"  Analyzed      : {batch['analyzed_count']}/{batch['headline_count']}")
    print(f"  🟢 Bullish    : {batch['bullish_count']}")
    print(f"  🔴 Bearish    : {batch['bearish_count']}")
    print(f"  ⚪ Neutral    : {batch['neutral_count']}")
    print(f"  Method        : {batch['method']}")

    # ── 3. full market sentiment (with mock Fear & Greed) ───────────
    print(f"\n🌍  Test 3: Market Sentiment (+ Fear & Greed)")
    print("-" * 50)

    mock_news = {
        "symbol": "BTCUSDT",
        "headlines": mock_hl,
        "fear_greed": {"value": 65, "label": "Greed", "timestamp": ""},
        "source_counts": {"test": len(mock_hl)},
        "total_headlines": len(mock_hl),
    }
    mkt = sa.get_market_sentiment(mock_news)

    print(f"  Sentiment     : {mkt['sentiment_score']:+.4f}  ({mkt['sentiment_label']})")
    print(f"  Confidence    : {mkt['confidence']:.4f}")
    print(f"  Fear & Greed  : {mkt['fear_greed_normalized']:+.4f}  ({mkt['fear_greed_label']})")
    print(f"  Method        : {mkt['combined_method']}")

    # ── 4. cache info ───────────────────────────────────────────────
    print(f"\n💾  Test 4: Cache Info")
    print("-" * 50)
    for k, v in sa.get_cache_info().items():
        print(f"  {k}: {v}")

    # ── 5. live news (optional) ─────────────────────────────────────
    print(f"\n🔴  Test 5: Live News Sentiment")
    print("-" * 50)
    try:
        from data.news_data import NewsData

        nd = NewsData()
        live = nd.get_all_news(symbol="BTCUSDT")

        if live and live.get("headlines"):
            lr = sa.get_market_sentiment(live)
            print(f"  Headlines     : {lr['news_sentiment']['analyzed_count']}")
            print(f"  Sentiment     : {lr['sentiment_score']:+.4f}  ({lr['sentiment_label']})")
            print(f"  Confidence    : {lr['confidence']:.4f}")

            dets = sorted(lr["news_sentiment"]["details"], key=lambda d: d["score"], reverse=True)
            if dets:
                print("  Top bullish:")
                for d in dets[:3]:
                    print(f"    [{d['score']:+.3f}] {d['text']}")
                print("  Top bearish:")
                for d in dets[-3:]:
                    print(f"    [{d['score']:+.3f}] {d['text']}")
        else:
            print("  No headlines available")
    except Exception as exc:
        print(f"  Skipped: {exc}")

    print(f"\n{'=' * 60}")
    print("✅ Sentiment analysis tests complete!")