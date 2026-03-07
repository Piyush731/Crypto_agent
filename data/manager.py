"""
data/manager.py - Data orchestration layer

Coordinates all data sources into a single dataset dict:
  - BinanceData  → OHLCV (multi-TF), funding rate, open interest, ticker, reference assets
  - NewsData     → headlines from RSS / Reddit, Fear & Greed index
  - SentimentAnalyzer → FinBERT scores on headlines, combined market sentiment

Output consumed by features/builder.py and analysis/signal_engine.py.
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from core.logger import get_logger
from data.binance_data import BinanceData
from data.news_data import NewsData
from data.sentiment import SentimentAnalyzer
import config

logger = get_logger("data.manager")


class DataManager:
    """
    Single entry-point for the entire data layer.

    Usage::

        dm = DataManager()
        dataset = dm.get_full_dataset("BTCUSDT")
        # dataset['ohlcv']['1h']  → DataFrame
        # dataset['sentiment']    → market sentiment dict
        # dataset['data_quality'] → {valid, score, issues, warnings}
    """

    def __init__(self):
        self.binance = BinanceData()
        self.news = NewsData()
        self.sentiment = SentimentAnalyzer()

        self.trading_pairs: List[str] = getattr(config, "TRADING_PAIRS", ["BTCUSDT"])
        self.reference_assets: List[str] = getattr(
            config, "REFERENCE_ASSETS", ["BTCUSDT", "ETHUSDT"]
        )

        # ── Handle TIMEFRAMES as dict OR list ──────────────────────
        tf_config = getattr(config, "TIMEFRAMES", ["1h", "4h", "1d"])
        if isinstance(tf_config, dict):
            # {"entry": "1h", "swing": "4h", "macro": "1d"} → ["1h", "4h", "1d"]
            self.timeframes: List[str] = list(tf_config.values())
            self._tf_roles: Dict[str, str] = tf_config  # keep role mapping
            logger.info(f"TIMEFRAMES (dict): roles={tf_config} → intervals={self.timeframes}")
        else:
            self.timeframes = list(tf_config)
            self._tf_roles = {}
            logger.info(f"TIMEFRAMES (list): {self.timeframes}")

        # Validate they are real Binance intervals
        valid_intervals = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
        for tf in self.timeframes:
            if tf not in valid_intervals:
                logger.error(f"Invalid Binance interval: '{tf}' — will fail API calls!")

        logger.info("DataManager initialized")

    # ═══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════════════

    def get_full_dataset(
        self,
        symbol: str,
        use_cache: bool = True,
        include_news: bool = True,
    ) -> Dict:
        """
        Fetch **everything** needed for one symbol's analysis pipeline.

        Parameters
        ----------
        symbol : str
            e.g. ``"BTCUSDT"``
        use_cache : bool
            Pass through to individual data sources.
        include_news : bool
            If *False*, skip news + sentiment (faster for training loops).

        Returns
        -------
        dict
            symbol, fetched_at,
            ohlcv          – ``{tf: DataFrame}``
            reference_data – ``{sym: DataFrame}``
            funding, open_interest, ticker,
            news, sentiment,
            data_quality   – ``{valid, score, issues, warnings}``
            fetch_errors   – ``List[str]``
            fetch_time_seconds – float
        """
        start = time.time()
        errors: List[str] = []

        result: Dict = {
            "symbol": symbol,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "ohlcv": {},
            "reference_data": {},
            "funding": None,
            "open_interest": None,
            "ticker": None,
            "news": None,
            "sentiment": None,
            "data_quality": {},
            "fetch_errors": [],
            "fetch_time_seconds": 0.0,
        }

        logger.info(f"{'─'*50}")
        logger.info(f"[{symbol}] Fetching full dataset …")

        # ── 1. OHLCV (multi-timeframe) ─────────────────────────────
        result["ohlcv"], errs = self._fetch_ohlcv(symbol, use_cache)
        errors.extend(errs)

        # ── 2. Funding rate ────────────────────────────────────────
        result["funding"], errs = self._fetch_safe(
            lambda: self.binance.get_funding_rate(symbol, use_cache=use_cache),
            "funding rate",
            symbol,
        )
        errors.extend(errs)

        # ── 3. Open interest ──────────────────────────────────────
        result["open_interest"], errs = self._fetch_safe(
            lambda: self.binance.get_open_interest(symbol, use_cache=use_cache),
            "open interest",
            symbol,
        )
        errors.extend(errs)

        # ── 4. Ticker ─────────────────────────────────────────────
        result["ticker"], errs = self._fetch_safe(
            lambda: self.binance.get_ticker(symbol, use_cache=use_cache),
            "ticker",
            symbol,
        )
        errors.extend(errs)

        # ── 5. Reference assets (BTC / ETH for cross-correlation) ─
        result["reference_data"], errs = self._fetch_reference(symbol, use_cache)
        errors.extend(errs)

        # ── 6. News + Sentiment ───────────────────────────────────
        if include_news:
            result["news"], result["sentiment"], errs = self._fetch_news_sentiment(
                symbol, use_cache
            )
            errors.extend(errs)

        # ── 7. Quality assessment ─────────────────────────────────
        result["fetch_errors"] = errors
        result["data_quality"] = self._assess_quality(result)
        result["fetch_time_seconds"] = round(time.time() - start, 2)

        q = result["data_quality"]
        logger.info(
            f"[{symbol}] Dataset complete in {result['fetch_time_seconds']}s │ "
            f"quality={q['score']}/100 │ valid={q['valid']} │ "
            f"errors={len(errors)} warnings={len(q['warnings'])}"
        )
        return result

    def get_multi_pair_data(
        self,
        symbols: Optional[List[str]] = None,
        use_cache: bool = True,
        include_news: bool = True,
    ) -> Dict[str, Dict]:
        """
        Run :meth:`get_full_dataset` for every symbol.

        Returns ``{symbol: dataset_dict}``.
        """
        if symbols is None:
            symbols = self.trading_pairs

        results: Dict[str, Dict] = {}
        for i, sym in enumerate(symbols, 1):
            logger.info(f"{'═'*50}")
            logger.info(f"[{i}/{len(symbols)}] Processing {sym}")
            results[sym] = self.get_full_dataset(
                sym, use_cache=use_cache, include_news=include_news
            )
            if i < len(symbols):
                time.sleep(0.5)  # rate-limit courtesy

        # summary
        valid = sum(1 for d in results.values() if d["data_quality"]["valid"])
        logger.info(f"{'═'*50}")
        logger.info(f"Multi-pair complete: {valid}/{len(symbols)} valid datasets")
        return results

    def validate_data(self, dataset: Dict) -> Dict:
        """
        Public wrapper around quality assessment.

        Returns
        -------
        dict
            valid    : bool – True if score >= 40 and no critical issues
            score    : int  – 0-100
            issues   : list – critical problems
            warnings : list – non-fatal notes
        """
        return self._assess_quality(dataset)

    def get_status(self) -> Dict:
        """Quick connectivity check for every data source."""
        status: Dict = {
            "binance": False,
            "news": False,
            "sentiment": False,
            "all_ok": False,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

        # Binance
        try:
            status["binance"] = self.binance.test_connection()
        except Exception as exc:
            logger.warning(f"Binance check failed: {exc}")

        # News (Fear & Greed is the fastest probe)
        try:
            fg = self.news.get_fear_greed_index()
            status["news"] = fg is not None
        except Exception as exc:
            logger.warning(f"News check failed: {exc}")

        # Sentiment (keyword fallback always works)
        try:
            probe = self.sentiment.analyze_headline("Bitcoin price rises")
            status["sentiment"] = probe.get("score") is not None
        except Exception as exc:
            logger.warning(f"Sentiment check failed: {exc}")

        status["all_ok"] = all(
            status[k] for k in ("binance", "news", "sentiment")
        )
        return status

    def get_cache_info(self) -> Dict:
        """Aggregate cache stats from all sub-modules."""
        info: Dict = {}
        try:
            info["binance"] = self.binance.get_cache_info()
        except Exception:
            info["binance"] = {}
        try:
            info["sentiment"] = self.sentiment.get_cache_info()
        except Exception:
            info["sentiment"] = {}
        return info

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear caches across all data sources."""
        try:
            self.binance.clear_cache(symbol)
        except Exception:
            pass
        try:
            self.news.clear_cache()
        except Exception:
            pass
        try:
            self.sentiment.clear_cache()
        except Exception:
            pass
        tag = f" for {symbol}" if symbol else ""
        logger.info(f"All caches cleared{tag}")

    # ═══════════════════════════════════════════════════════════════
    #  INTERNAL HELPERS
    # ═══════════════════════════════════════════════════════════════

    def _fetch_safe(self, fn, label: str, symbol: str):
        """
        Call *fn()* with error isolation.

        Returns ``(result_or_None, error_list)``.
        """
        try:
            data = fn()
            if data is not None:
                logger.info(f"  [{symbol}] ✅ {label}")
            else:
                logger.warning(f"  [{symbol}] ⚠️  {label}: no data")
            return data, []
        except Exception as exc:
            logger.error(f"  [{symbol}] ❌ {label}: {exc}")
            return None, [f"{label}: {exc}"]

    def _fetch_ohlcv(self, symbol: str, use_cache: bool):
        """Fetch multi-TF OHLCV with per-timeframe error isolation."""
        errors: List[str] = []
        ohlcv: Dict[str, pd.DataFrame] = {}

        try:
            # Pass self.timeframes which are real Binance intervals ["1h","4h","1d"]
            raw = self.binance.get_multi_tf_ohlcv(
                symbol, timeframes=self.timeframes, use_cache=use_cache
            )
            if raw:
                ohlcv = raw
                for tf, df in ohlcv.items():
                    if df is not None and not df.empty:
                        logger.info(f"  [{symbol}] ✅ OHLCV {tf}: {len(df)} candles")
                    else:
                        logger.warning(f"  [{symbol}] ⚠️  OHLCV {tf}: empty")
                        errors.append(f"OHLCV {tf}: empty")
            else:
                errors.append("OHLCV: get_multi_tf_ohlcv returned empty")
                logger.error(f"  [{symbol}] ❌ OHLCV: no data returned")
        except Exception as exc:
            errors.append(f"OHLCV: {exc}")
            logger.error(f"  [{symbol}] ❌ OHLCV: {exc}")

        return ohlcv, errors

    def _fetch_reference(self, symbol: str, use_cache: bool):
        """Fetch reference-asset 1h OHLCV (BTC, ETH, …)."""
        errors: List[str] = []
        ref: Dict[str, pd.DataFrame] = {}

        try:
            raw = self.binance.get_reference_data(
                exclude_symbol=symbol, use_cache=use_cache
            )
            if raw:
                ref = raw
                for sym, df in ref.items():
                    logger.info(f"  [{symbol}] ✅ Ref {sym}: {len(df)} candles")
            else:
                logger.warning(f"  [{symbol}] ⚠️  Reference data: empty")
        except Exception as exc:
            errors.append(f"Reference data: {exc}")
            logger.warning(f"  [{symbol}] ⚠️  Reference data: {exc}")

        return ref, errors

    def _fetch_news_sentiment(self, symbol: str, use_cache: bool):
        """Fetch news headlines, run sentiment analysis."""
        errors: List[str] = []
        news_data = None
        sentiment_data = None

        # Headlines + Fear & Greed
        try:
            news_data = self.news.get_all_news(symbol=symbol, use_cache=use_cache)
            if news_data and news_data.get("headlines"):
                logger.info(
                    f"  [{symbol}] ✅ News: {news_data['total_headlines']} headlines"
                )
            else:
                logger.warning(f"  [{symbol}] ⚠️  News: no headlines")
        except Exception as exc:
            errors.append(f"News: {exc}")
            logger.warning(f"  [{symbol}] ⚠️  News: {exc}")

        # Sentiment scoring
        if news_data:
            try:
                sentiment_data = self.sentiment.get_market_sentiment(news_data)
                logger.info(
                    f"  [{symbol}] ✅ Sentiment: "
                    f"{sentiment_data['sentiment_label']} "
                    f"({sentiment_data['sentiment_score']:+.3f})"
                )
            except Exception as exc:
                errors.append(f"Sentiment: {exc}")
                logger.warning(f"  [{symbol}] ⚠️  Sentiment: {exc}")

        return news_data, sentiment_data, errors

    def _assess_quality(self, dataset: Dict) -> Dict:
        """
        Score dataset completeness on a 0-100 scale.

        Scoring deductions:
          - Missing OHLCV timeframe    → -15 each  (critical)
          - Too few candles            → -5
          - NaN in close prices        → -5
          - Stale data                 → -3
          - Missing funding            → -5
          - Missing ticker             → -3
          - Missing reference data     → -5
          - Missing sentiment          → -5
          - Each fetch error           → -3
        """
        issues: List[str] = []
        warnings: List[str] = []
        score = 100

        ohlcv = dataset.get("ohlcv", {})

        # ── OHLCV checks (most critical) ──────────────────────────
        if not ohlcv:
            issues.append("No OHLCV data at all")
            score -= 50
        else:
            min_candles = {"1h": 100, "4h": 50, "1d": 30}
            for tf in self.timeframes:
                df = ohlcv.get(tf)
                if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                    issues.append(f"Missing {tf} OHLCV")
                    score -= 15
                    continue

                needed = min_candles.get(tf, 50)
                n = len(df)
                if n < needed:
                    warnings.append(f"{tf}: {n}/{needed} candles (low)")
                    score -= 5

                # NaN check
                if "close" in df.columns and df["close"].isna().any():
                    nan_pct = df["close"].isna().mean() * 100
                    warnings.append(f"{tf}: {nan_pct:.1f}% NaN in close")
                    score -= 5

                # Freshness check
                try:
                    last_ts = df.index.max()
                    if pd.notna(last_ts):
                        now_utc = pd.Timestamp.now(tz="UTC")
                        if last_ts.tzinfo is None:
                            last_ts = last_ts.tz_localize("UTC")
                        age_h = (now_utc - last_ts).total_seconds() / 3600
                        max_age = {"1h": 3, "4h": 8, "1d": 48}
                        if age_h > max_age.get(tf, 48):
                            warnings.append(
                                f"{tf}: last candle {age_h:.1f}h old"
                            )
                            score -= 3
                except Exception:
                    pass

        # ── Supporting data ────────────────────────────────────────
        if dataset.get("funding") is None:
            warnings.append("No funding rate")
            score -= 5

        if dataset.get("ticker") is None:
            warnings.append("No ticker data")
            score -= 3

        ref = dataset.get("reference_data", {})
        if not ref:
            warnings.append("No reference asset data")
            score -= 5

        if dataset.get("sentiment") is None:
            warnings.append("No sentiment data")
            score -= 5

        # ── Fetch errors ───────────────────────────────────────────
        n_err = len(dataset.get("fetch_errors", []))
        if n_err:
            score -= n_err * 3

        score = max(0, min(100, score))
        valid = score >= 40 and len(issues) == 0

        return {
            "valid": valid,
            "score": score,
            "issues": issues,
            "warnings": warnings,
            "error_count": n_err,
        }


# ═══════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("DATA MANAGER TEST")
    print("=" * 60)

    dm = DataManager()

    # ── 0. Show resolved timeframes ─────────────────────────────
    print(f"\n⏱️  Resolved Timeframes: {dm.timeframes}")
    if dm._tf_roles:
        print(f"   Role mapping: {dm._tf_roles}")
    print()

    # ── 1. Status check ─────────────────────────────────────────
    print("🔌  Test 1: Source Connectivity")
    print("-" * 50)
    status = dm.get_status()
    for k, v in status.items():
        icon = "✅" if v else "❌"
        if k == "checked_at":
            print(f"  ⏰ {k}: {v}")
        else:
            print(f"  {icon} {k}: {v}")

    if not status["binance"]:
        print("\n⛔ Binance unreachable – cannot continue")
        exit(1)

    # ── 2. Full dataset for BTCUSDT ─────────────────────────────
    print(f"\n📦  Test 2: Full Dataset (BTCUSDT)")
    print("-" * 50)
    ds = dm.get_full_dataset("BTCUSDT", use_cache=True, include_news=True)

    print(f"\n  Symbol        : {ds['symbol']}")
    print(f"  Fetched at    : {ds['fetched_at']}")
    print(f"  Fetch time    : {ds['fetch_time_seconds']}s")

    # OHLCV
    print(f"\n  📊 OHLCV:")
    for tf, df in ds["ohlcv"].items():
        if df is not None and not df.empty:
            print(
                f"    {tf:4s}: {len(df):4d} candles │ "
                f"{df.index.min()} → {df.index.max()} │ "
                f"last close: {df['close'].iloc[-1]:.2f}"
            )
        else:
            print(f"    {tf:4s}: EMPTY")

    if not ds["ohlcv"]:
        print("    ⚠️  NO OHLCV DATA — check TIMEFRAMES in config.py!")

    # Reference data
    print(f"\n  📈 Reference Assets:")
    for sym, df in ds.get("reference_data", {}).items():
        if df is not None and not df.empty:
            print(f"    {sym:10s}: {len(df)} candles, last: {df['close'].iloc[-1]:.2f}")
        else:
            print(f"    {sym:10s}: EMPTY")

    # Funding
    fr = ds.get("funding")
    if fr:
        print(f"\n  💰 Funding Rate:")
        print(f"    Current     : {fr.get('current_rate', 'N/A')}")
        print(f"    Annualized  : {fr.get('annualized_rate', 'N/A')}")
    else:
        print(f"\n  💰 Funding Rate: N/A")

    # Open interest
    oi = ds.get("open_interest")
    if oi:
        print(f"  📐 Open Interest: {oi.get('open_interest', 'N/A')}")
    else:
        print(f"  📐 Open Interest: N/A")

    # Ticker
    tk = ds.get("ticker")
    if tk:
        print(f"\n  🏷️  Ticker:")
        print(f"    Price       : {tk.get('last_price', 'N/A')}")
        print(f"    24h Change  : {tk.get('price_change_pct', 'N/A')}%")
        print(f"    24h Volume  : {tk.get('quote_volume_24h', 'N/A')}")
    else:
        print(f"\n  🏷️  Ticker: N/A")

    # Sentiment
    sent = ds.get("sentiment")
    if sent:
        ns = sent.get("news_sentiment", {})
        print(f"\n  🧠 Sentiment:")
        print(f"    Score       : {sent['sentiment_score']:+.4f}")
        print(f"    Label       : {sent['sentiment_label']}")
        print(f"    Confidence  : {sent['confidence']:.4f}")
        print(f"    Headlines   : {ns.get('analyzed_count', 0)}")
        print(f"    Method      : {sent['combined_method']}")
        fg = sent.get("fear_greed")
        if fg:
            print(
                f"    Fear & Greed: {fg.get('value', '?')} "
                f"({fg.get('label', '?')})"
            )
    else:
        print(f"\n  🧠 Sentiment: N/A")

    # Data quality
    q = ds["data_quality"]
    print(f"\n  ✅ Data Quality:")
    print(f"    Score       : {q['score']}/100")
    print(f"    Valid       : {q['valid']}")
    if q["issues"]:
        print(f"    Issues      :")
        for iss in q["issues"]:
            print(f"      ❌ {iss}")
    if q["warnings"]:
        print(f"    Warnings    :")
        for w in q["warnings"]:
            print(f"      ⚠️  {w}")
    if ds["fetch_errors"]:
        print(f"    Fetch Errors:")
        for e in ds["fetch_errors"]:
            print(f"      ❌ {e}")

    # ── 3. Validate separately ──────────────────────────────────
    print(f"\n🔍  Test 3: Validate Dataset")
    print("-" * 50)
    v = dm.validate_data(ds)
    print(f"  Valid: {v['valid']}  Score: {v['score']}/100")

    # ── 4. Cache info ───────────────────────────────────────────
    print(f"\n💾  Test 4: Cache Info")
    print("-" * 50)
    ci = dm.get_cache_info()
    for source, info in ci.items():
        print(f"  {source}:")
        if isinstance(info, dict):
            for k, v2 in info.items():
                print(f"    {k}: {v2}")
        else:
            print(f"    {info}")

    # ── 5. Dataset without news (training mode) ────────────────
    print(f"\n⚡  Test 5: Dataset without news (faster)")
    print("-" * 50)
    ds_fast = dm.get_full_dataset("ETHUSDT", use_cache=True, include_news=False)
    print(f"  Symbol   : {ds_fast['symbol']}")
    print(f"  Time     : {ds_fast['fetch_time_seconds']}s")
    print(f"  OHLCV TFs: {list(ds_fast['ohlcv'].keys())}")
    print(f"  News     : {ds_fast['news']}")
    print(f"  Sentiment: {ds_fast['sentiment']}")
    print(f"  Quality  : {ds_fast['data_quality']['score']}/100")

    # ── 6. Clear cache ──────────────────────────────────────────
    print(f"\n🧹  Test 6: Clear Cache")
    print("-" * 50)
    dm.clear_cache()
    print("  Done")

    print(f"\n{'=' * 60}")
    print("✅ DataManager tests complete!")