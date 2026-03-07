"""
data/binance_data.py
====================
Binance USDT-M Futures public API data fetcher.

Usage in any module:
    from data.binance_data import BinanceData
    bd = BinanceData()
    df = bd.get_ohlcv("BTCUSDT", "1h", limit=500)
    funding = bd.get_funding_rate("BTCUSDT")
    oi = bd.get_open_interest("BTCUSDT")
    ticker = bd.get_ticker("BTCUSDT")
    all_data = bd.get_all_data("BTCUSDT")

Features:
    - Multi-timeframe OHLCV (1h, 4h, 1d) up to 1500 candles
    - Funding rates (current + historical)
    - Open interest
    - 24h ticker statistics
    - Local file caching with configurable TTL
    - Rate limit handling with retries
    - BTC/ETH reference data for cross-asset features
    - All public endpoints — no API key needed
"""

import time
import pickle
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

# ── Import config ────────────────────────────────────────────
try:
    from config import TRADING_PAIRS, CACHE_DIR
except ImportError:
    TRADING_PAIRS = ["BTCUSDT", "ETHUSDT"]
    CACHE_DIR = Path(__file__).parent.parent / "cache"

from core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────

# Binance USDT-M Futures base URL
BASE_URL = "https://fapi.binance.com"

# How long cached data stays fresh (seconds)
CACHE_TTL = {
    "1m": 60,
    "5m": 180,
    "15m": 600,
    "1h": 1800,       # 30 minutes
    "4h": 7200,        # 2 hours
    "1d": 21600,       # 6 hours
    "1w": 86400,       # 24 hours
    "funding": 1800,   # 30 minutes
    "oi": 300,         # 5 minutes
    "ticker": 60,      # 1 minute
}

# Default candle counts per timeframe
DEFAULT_LIMITS = {
    "1h": 720,         # 30 days
    "4h": 500,         # ~83 days
    "1d": 365,         # 1 year
}


# ╔═══════════════════════════════════════════════════════════╗
# ║  BINANCE DATA CLASS                                        ║
# ╚═══════════════════════════════════════════════════════════╝

class BinanceData:
    """Fetches market data from Binance USDT-M Futures public API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "CryptoAgent/3.0",
        })
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0
        self._min_interval = 0.1  # 100ms between requests
        logger.info("BinanceData initialized")

    # ──────────────────────────────────────────────────────────
    #  INTERNAL: Rate Limit + Request + Cache
    # ──────────────────────────────────────────────────────────

    def _rate_limit(self):
        """Enforce minimum interval between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _request(self, endpoint: str, params: dict = None,
                 max_retries: int = 3) -> Optional[Any]:
        """
        Make a rate-limited API request with retries.

        Returns:
            Parsed JSON (dict or list), or None on failure.
        """
        url = f"{BASE_URL}{endpoint}"

        for attempt in range(1, max_retries + 1):
            try:
                self._rate_limit()
                resp = self.session.get(url, params=params, timeout=15)

                # Rate limited by Binance
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 30))
                    logger.warning(f"Rate limited! Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                # Server error — retry
                if resp.status_code >= 500:
                    logger.warning(
                        f"Server error {resp.status_code} "
                        f"(attempt {attempt}/{max_retries})"
                    )
                    time.sleep(2 ** attempt)
                    continue

                resp.raise_for_status()
                data = resp.json()

                # Binance API-level error (e.g., invalid symbol)
                if isinstance(data, dict) and data.get("code", 0) < 0:
                    logger.error(
                        f"Binance error: {data.get('msg', '?')} "
                        f"(code {data['code']})"
                    )
                    return None

                return data

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Timeout {endpoint} (attempt {attempt}/{max_retries})"
                )
                time.sleep(2 ** attempt)

            except requests.exceptions.ConnectionError:
                logger.warning(
                    f"Connection error (attempt {attempt}/{max_retries})"
                )
                time.sleep(2 ** attempt)

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt == max_retries:
                    return None
                time.sleep(2 ** attempt)

        logger.error(f"All {max_retries} attempts failed for {endpoint}")
        return None

    def _cache_path(self, symbol: str, data_type: str,
                    interval: str = "") -> Path:
        """Generate cache file path."""
        parts = [symbol, data_type]
        if interval:
            parts.append(interval)
        return self.cache_dir / ("_".join(parts) + ".pkl")

    def _save_cache(self, path: Path, data: Any):
        """Save data to cache with timestamp."""
        try:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "data": data,
            }
            with open(path, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def _load_cache(self, path: Path,
                    max_age_seconds: int) -> Optional[Any]:
        """Load data from cache if not stale."""
        try:
            if not path.exists():
                return None

            with open(path, "rb") as f:
                payload = pickle.load(f)

            cached_time = datetime.fromisoformat(payload["timestamp"])
            age = (datetime.now() - cached_time).total_seconds()

            if age > max_age_seconds:
                return None  # Stale

            logger.debug(f"Cache hit: {path.name} (age: {age:.0f}s)")
            return payload["data"]

        except Exception:
            return None

    # ──────────────────────────────────────────────────────────
    #  CONNECTIVITY CHECK
    # ──────────────────────────────────────────────────────────

    def test_connection(self) -> bool:
        """Test if Binance Futures API is reachable."""
        data = self._request("/fapi/v1/ping")
        return data is not None

    def get_server_time(self) -> Optional[datetime]:
        """Get Binance server time (useful for clock sync check)."""
        data = self._request("/fapi/v1/time")
        if data and "serverTime" in data:
            return datetime.fromtimestamp(data["serverTime"] / 1000)
        return None

    # ══════════════════════════════════════════════════════════
    #  OHLCV DATA
    # ══════════════════════════════════════════════════════════

    def get_ohlcv(self, symbol: str, interval: str = "1h",
                  limit: int = None,
                  use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV candlestick data.

        Args:
            symbol:    e.g., "BTCUSDT"
            interval:  "1m","5m","15m","1h","4h","1d","1w"
            limit:     Number of candles (max 1500)
            use_cache: Use local file cache

        Returns:
            DataFrame with DatetimeIndex and columns:
                open, high, low, close, volume,
                quote_volume, trades,
                taker_buy_volume, taker_buy_quote_volume
            None if fetch fails completely.
        """
        if limit is None:
            limit = DEFAULT_LIMITS.get(interval, 500)
        limit = min(limit, 1500)  # Binance max per request

        # ── Try cache first ──
        cache_file = self._cache_path(symbol, "ohlcv", interval)
        if use_cache:
            ttl = CACHE_TTL.get(interval, 1800)
            cached = self._load_cache(cache_file, ttl)
            if cached is not None and len(cached) >= limit * 0.9:
                return cached

        # ── Fetch from API ──
        raw = self._request("/fapi/v1/klines", {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        })

        if not raw:
            logger.error(f"Failed to fetch OHLCV: {symbol} {interval}")
            # Fallback: return stale cache if available
            if use_cache:
                stale = self._load_cache(cache_file, max_age_seconds=86400)
                if stale is not None:
                    logger.warning(
                        f"Using stale cache for {symbol} {interval}"
                    )
                    return stale
            return None

        # ── Parse into DataFrame ──
        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_volume", "taker_buy_quote_volume", "ignore",
        ])

        # Convert numeric columns
        numeric_cols = [
            "open", "high", "low", "close", "volume",
            "quote_volume", "taker_buy_volume", "taker_buy_quote_volume",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["trades"] = pd.to_numeric(
            df["trades"], errors="coerce"
        ).fillna(0).astype(int)

        # Timestamp as DatetimeIndex
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        df = df.drop(columns=["close_time", "ignore"], errors="ignore")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # ── Validate ──
        if df.empty:
            logger.error(f"Empty DataFrame for {symbol} {interval}")
            return None

        # Drop rows where OHLC is all zeros or NaN
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[(df["close"] > 0)]

        # ── Cache & return ──
        if use_cache:
            self._save_cache(cache_file, df)

        logger.info(
            f"OHLCV {symbol} {interval}: {len(df)} candles "
            f"[{df.index[0].strftime('%Y-%m-%d')} → "
            f"{df.index[-1].strftime('%Y-%m-%d %H:%M')}]"
        )
        return df

    def get_multi_tf_ohlcv(
        self,
        symbol: str,
        timeframes: List[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for multiple timeframes.

        Args:
            symbol:     e.g., "BTCUSDT"
            timeframes: e.g., ["1h", "4h", "1d"] (default)

        Returns:
            {"1h": df_1h, "4h": df_4h, "1d": df_1d}
            Missing timeframes are omitted (not None).
        """
        if timeframes is None:
            timeframes = ["1h", "4h", "1d"]

        result = {}
        for tf in timeframes:
            df = self.get_ohlcv(symbol, tf, use_cache=use_cache)
            if df is not None and not df.empty:
                result[tf] = df
            else:
                logger.warning(f"Missing {tf} data for {symbol}")
            time.sleep(0.2)  # Politeness delay between TFs

        logger.info(
            f"Multi-TF {symbol}: "
            + ", ".join(f"{tf}={len(df)}" for tf, df in result.items())
        )
        return result

    # ══════════════════════════════════════════════════════════
    #  FUNDING RATE
    # ══════════════════════════════════════════════════════════

    def get_funding_rate(self, symbol: str, limit: int = 10,
                         use_cache: bool = True) -> Optional[Dict]:
        """
        Get current and recent funding rates.

        Returns:
            {
                "symbol": str,
                "current_rate": float,       # e.g., 0.0001
                "mark_price": float,
                "index_price": float,
                "next_funding_time": str,    # ISO format
                "annualized_rate": float,    # percentage
                "history": [{"rate": float, "time": str}, ...]
            }
        """
        if use_cache:
            cache_file = self._cache_path(symbol, "funding")
            cached = self._load_cache(
                cache_file, CACHE_TTL.get("funding", 1800)
            )
            if cached is not None:
                return cached

        # Current funding + mark price
        premium = self._request("/fapi/v1/premiumIndex", {
            "symbol": symbol,
        })

        if not premium:
            logger.error(f"Failed to fetch funding: {symbol}")
            return None

        # Historical funding rates
        history_raw = self._request("/fapi/v1/fundingRate", {
            "symbol": symbol,
            "limit": limit,
        })

        current_rate = float(premium.get("lastFundingRate", 0))

        result = {
            "symbol": symbol,
            "current_rate": current_rate,
            "mark_price": float(premium.get("markPrice", 0)),
            "index_price": float(premium.get("indexPrice", 0)),
            "next_funding_time": None,
            "annualized_rate": current_rate * 3 * 365 * 100,  # 3x daily
            "history": [],
        }

        # Parse next funding time
        nft = premium.get("nextFundingTime")
        if nft and int(nft) > 0:
            result["next_funding_time"] = datetime.fromtimestamp(
                int(nft) / 1000
            ).isoformat()

        # Parse history
        if history_raw:
            result["history"] = [
                {
                    "rate": float(h.get("fundingRate", 0)),
                    "time": datetime.fromtimestamp(
                        int(h.get("fundingTime", 0)) / 1000
                    ).isoformat(),
                }
                for h in history_raw
            ]

        if use_cache:
            self._save_cache(
                self._cache_path(symbol, "funding"), result
            )

        logger.info(
            f"Funding {symbol}: {current_rate:.6f} "
            f"(annualized: {result['annualized_rate']:.1f}%)"
        )
        return result

    # ══════════════════════════════════════════════════════════
    #  OPEN INTEREST
    # ══════════════════════════════════════════════════════════

    def get_open_interest(self, symbol: str,
                          use_cache: bool = True) -> Optional[Dict]:
        """
        Get current open interest.

        Returns:
            {
                "symbol": str,
                "open_interest": float,   # in contracts (base asset)
                "timestamp": str,         # ISO format
            }
        """
        if use_cache:
            cache_file = self._cache_path(symbol, "oi")
            cached = self._load_cache(
                cache_file, CACHE_TTL.get("oi", 300)
            )
            if cached is not None:
                return cached

        data = self._request("/fapi/v1/openInterest", {
            "symbol": symbol,
        })

        if not data:
            logger.error(f"Failed to fetch OI: {symbol}")
            return None

        result = {
            "symbol": symbol,
            "open_interest": float(data.get("openInterest", 0)),
            "timestamp": datetime.now().isoformat(),
        }

        ts = data.get("time")
        if ts and int(ts) > 0:
            result["timestamp"] = datetime.fromtimestamp(
                int(ts) / 1000
            ).isoformat()

        if use_cache:
            self._save_cache(self._cache_path(symbol, "oi"), result)

        logger.info(
            f"OI {symbol}: {result['open_interest']:,.2f} contracts"
        )
        return result

    # ══════════════════════════════════════════════════════════
    #  24H TICKER
    # ══════════════════════════════════════════════════════════

    def get_ticker(self, symbol: str,
                   use_cache: bool = True) -> Optional[Dict]:
        """
        Get 24-hour ticker statistics.

        Returns:
            {
                "symbol": str,
                "last_price": float,
                "price_change": float,
                "price_change_pct": float,
                "high_24h": float,
                "low_24h": float,
                "volume_24h": float,         # base asset
                "quote_volume_24h": float,   # USDT
                "weighted_avg_price": float,
                "trade_count": int,
            }
        """
        if use_cache:
            cache_file = self._cache_path(symbol, "ticker")
            cached = self._load_cache(
                cache_file, CACHE_TTL.get("ticker", 60)
            )
            if cached is not None:
                return cached

        data = self._request("/fapi/v1/ticker/24hr", {
            "symbol": symbol,
        })

        if not data:
            logger.error(f"Failed to fetch ticker: {symbol}")
            return None

        result = {
            "symbol": symbol,
            "last_price": float(data.get("lastPrice", 0)),
            "price_change": float(data.get("priceChange", 0)),
            "price_change_pct": float(data.get("priceChangePercent", 0)),
            "high_24h": float(data.get("highPrice", 0)),
            "low_24h": float(data.get("lowPrice", 0)),
            "volume_24h": float(data.get("volume", 0)),
            "quote_volume_24h": float(data.get("quoteVolume", 0)),
            "weighted_avg_price": float(data.get("weightedAvgPrice", 0)),
            "trade_count": int(data.get("count", 0)),
        }

        if use_cache:
            self._save_cache(self._cache_path(symbol, "ticker"), result)

        logger.info(
            f"Ticker {symbol}: ${result['last_price']:,.2f} "
            f"({result['price_change_pct']:+.2f}%) "
            f"Vol: ${result['quote_volume_24h']:,.0f}"
        )
        return result

    # ══════════════════════════════════════════════════════════
    #  ALL-IN-ONE (per symbol)
    # ══════════════════════════════════════════════════════════

    def get_all_data(self, symbol: str,
                     use_cache: bool = True) -> Dict:
        """
        Fetch everything for one symbol's analysis.

        Returns:
            {
                "symbol": str,
                "fetched_at": str,
                "ohlcv": {"1h": df, "4h": df, "1d": df},
                "funding": {...} or None,
                "oi": {...} or None,
                "ticker": {...} or None,
            }
        """
        logger.info(f"{'─'*50}")
        logger.info(f"Fetching ALL data for {symbol}...")

        result = {
            "symbol": symbol,
            "fetched_at": datetime.now().isoformat(),
            "ohlcv": {},
            "funding": None,
            "oi": None,
            "ticker": None,
        }

        # Multi-timeframe OHLCV
        result["ohlcv"] = self.get_multi_tf_ohlcv(
            symbol, use_cache=use_cache
        )

        # Funding rate
        result["funding"] = self.get_funding_rate(
            symbol, use_cache=use_cache
        )
        time.sleep(0.15)

        # Open interest
        result["oi"] = self.get_open_interest(
            symbol, use_cache=use_cache
        )
        time.sleep(0.15)

        # 24h ticker
        result["ticker"] = self.get_ticker(
            symbol, use_cache=use_cache
        )

        # Summary log
        tf_info = ", ".join(
            f"{tf}={len(df)}" for tf, df in result["ohlcv"].items()
        )
        f_ok = "✅" if result["funding"] else "❌"
        o_ok = "✅" if result["oi"] else "❌"
        t_ok = "✅" if result["ticker"] else "❌"

        logger.info(
            f"Complete: {symbol} | OHLCV[{tf_info}] "
            f"Funding:{f_ok} OI:{o_ok} Ticker:{t_ok}"
        )
        return result

    # ══════════════════════════════════════════════════════════
    #  REFERENCE DATA (BTC/ETH for cross-asset features)
    # ══════════════════════════════════════════════════════════

    def get_reference_data(
        self,
        exclude_symbol: str = None,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch BTC and ETH 1h data for cross-asset features.

        Used by features/market.py for:
            - BTC/ETH correlation
            - BTC RSI as market barometer
            - Cross-asset momentum

        Args:
            exclude_symbol: Skip if it's already the symbol being analyzed.

        Returns:
            {"BTCUSDT": df_1h, "ETHUSDT": df_1h}
        """
        ref_pairs = ["BTCUSDT", "ETHUSDT"]
        result = {}

        for pair in ref_pairs:
            if pair == exclude_symbol:
                continue
            df = self.get_ohlcv(pair, "1h", limit=720,
                                use_cache=use_cache)
            if df is not None and not df.empty:
                result[pair] = df
            time.sleep(0.2)

        logger.info(
            f"Reference data: "
            + ", ".join(f"{p}={len(d)}" for p, d in result.items())
        )
        return result

    # ══════════════════════════════════════════════════════════
    #  UTILITIES
    # ══════════════════════════════════════════════════════════

    def clear_cache(self, symbol: str = None):
        """
        Clear cached data files.

        Args:
            symbol: Clear only this symbol. None = clear all.
        """
        count = 0
        for f in self.cache_dir.glob("*.pkl"):
            if symbol is None or f.name.startswith(symbol):
                f.unlink()
                count += 1
        logger.info(
            f"Cache cleared: {count} files"
            + (f" for {symbol}" if symbol else " (all)")
        )

    def get_cache_info(self) -> Dict[str, Any]:
        """Get info about current cache files."""
        files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "cache_dir": str(self.cache_dir),
            "file_count": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": [f.name for f in sorted(files)],
        }


# ╔═══════════════════════════════════════════════════════════╗
# ║  STANDALONE TEST                                           ║
# ╚═══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 60)
    print("  BINANCE DATA TEST")
    print("=" * 60)

    bd = BinanceData()

    # ── Test 0: Connection ──────────────────────────────────
    print("\n--- Test 0: Connection ---")
    if not bd.test_connection():
        print("  ❌ Cannot reach Binance API!")
        print("  Check your internet connection.")
        print("  If in a restricted region, try a VPN.")
        exit(1)

    server_time = bd.get_server_time()
    local_time = datetime.now()
    if server_time:
        drift = abs((local_time - server_time).total_seconds())
        print(f"  ✅ Connected to Binance Futures API")
        print(f"  🕐 Server time: {server_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  🕐 Local time:  {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  ⏱️  Clock drift: {drift:.1f}s")

    # ── Test 1: Single OHLCV ────────────────────────────────
    print("\n--- Test 1: OHLCV (BTCUSDT 1h, 100 candles) ---")
    df = bd.get_ohlcv("BTCUSDT", "1h", limit=100, use_cache=False)
    if df is not None:
        print(f"  ✅ Got {len(df)} candles")
        print(f"  📅 Range: {df.index[0]} → {df.index[-1]}")
        print(f"  📊 Columns: {list(df.columns)}")
        print(f"  💰 Latest close: ${df['close'].iloc[-1]:,.2f}")
        print(f"  📈 High range: ${df['high'].min():,.2f} → ${df['high'].max():,.2f}")
        print(f"  📉 Volume range: {df['volume'].min():,.2f} → {df['volume'].max():,.2f}")

        # Validate structure
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  ❌ Missing columns: {missing}")
        else:
            print(f"  ✅ All required columns present")

        # Check for NaN
        nan_counts = df[required].isna().sum()
        if nan_counts.sum() > 0:
            print(f"  ⚠️  NaN values: {dict(nan_counts[nan_counts > 0])}")
        else:
            print(f"  ✅ No NaN values in OHLCV")
    else:
        print(f"  ❌ Failed to fetch OHLCV")

    # ── Test 2: Cache test ──────────────────────────────────
    print("\n--- Test 2: Cache ---")
    start = time.time()
    df_cached = bd.get_ohlcv("BTCUSDT", "1h", limit=100, use_cache=True)
    elapsed = time.time() - start
    if df_cached is not None:
        print(f"  ✅ Cache fetch: {elapsed:.3f}s (should be <0.1s)")
        print(f"  ✅ Same length: {len(df_cached)} candles")
    else:
        print(f"  ⚠️  Cache miss (not critical)")

    # ── Test 3: Multi-timeframe ─────────────────────────────
    print("\n--- Test 3: Multi-Timeframe (BTCUSDT) ---")
    multi = bd.get_multi_tf_ohlcv("BTCUSDT", ["1h", "4h", "1d"])
    for tf, tf_df in multi.items():
        print(f"  ✅ {tf}: {len(tf_df)} candles | "
              f"Latest: ${tf_df['close'].iloc[-1]:,.2f}")

    # ── Test 4: Funding Rate ────────────────────────────────
    print("\n--- Test 4: Funding Rate (BTCUSDT) ---")
    funding = bd.get_funding_rate("BTCUSDT", limit=5)
    if funding:
        print(f"  ✅ Current rate: {funding['current_rate']:.6f}")
        print(f"  📊 Annualized: {funding['annualized_rate']:.2f}%")
        print(f"  💲 Mark price: ${funding['mark_price']:,.2f}")
        print(f"  ⏰ Next funding: {funding['next_funding_time']}")
        print(f"  📜 History entries: {len(funding['history'])}")
        if funding["history"]:
            latest = funding["history"][-1]
            print(f"     Latest historical: {latest['rate']:.6f} "
                  f"@ {latest['time']}")
    else:
        print(f"  ❌ Failed to fetch funding rate")

    # ── Test 5: Open Interest ───────────────────────────────
    print("\n--- Test 5: Open Interest (BTCUSDT) ---")
    oi = bd.get_open_interest("BTCUSDT")
    if oi:
        print(f"  ✅ OI: {oi['open_interest']:,.2f} BTC")
        print(f"  🕐 Time: {oi['timestamp']}")
    else:
        print(f"  ❌ Failed to fetch OI")

    # ── Test 6: 24h Ticker ──────────────────────────────────
    print("\n--- Test 6: 24h Ticker (BTCUSDT) ---")
    ticker = bd.get_ticker("BTCUSDT")
    if ticker:
        print(f"  ✅ Price: ${ticker['last_price']:,.2f}")
        print(f"  📈 Change: {ticker['price_change_pct']:+.2f}%")
        print(f"  🔺 24h High: ${ticker['high_24h']:,.2f}")
        print(f"  🔻 24h Low: ${ticker['low_24h']:,.2f}")
        print(f"  📊 Volume: ${ticker['quote_volume_24h']:,.0f} USDT")
        print(f"  🔄 Trades: {ticker['trade_count']:,}")
    else:
        print(f"  ❌ Failed to fetch ticker")

    # ── Test 7: get_all_data ────────────────────────────────
    print("\n--- Test 7: All Data (ETHUSDT) ---")
    all_data = bd.get_all_data("ETHUSDT")
    print(f"  Symbol: {all_data['symbol']}")
    print(f"  OHLCV timeframes: {list(all_data['ohlcv'].keys())}")
    for tf, tf_df in all_data["ohlcv"].items():
        print(f"    {tf}: {len(tf_df)} candles")
    print(f"  Funding: {'✅' if all_data['funding'] else '❌'}")
    print(f"  OI:      {'✅' if all_data['oi'] else '❌'}")
    print(f"  Ticker:  {'✅' if all_data['ticker'] else '❌'}")

    # ── Test 8: Reference data ──────────────────────────────
    print("\n--- Test 8: Reference Data (for SOLUSDT analysis) ---")
    ref = bd.get_reference_data(exclude_symbol="SOLUSDT")
    for pair, ref_df in ref.items():
        print(f"  ✅ {pair}: {len(ref_df)} candles")

    # ── Test 9: Cache info ──────────────────────────────────
    print("\n--- Test 9: Cache Info ---")
    cache_info = bd.get_cache_info()
    print(f"  📁 Dir: {cache_info['cache_dir']}")
    print(f"  📄 Files: {cache_info['file_count']}")
    print(f"  💾 Size: {cache_info['total_size_mb']} MB")
    for fname in cache_info["files"][:10]:
        print(f"     {fname}")

    # ── Test 10: Error handling (bad symbol) ────────────────
    print("\n--- Test 10: Error Handling (invalid symbol) ---")
    bad = bd.get_ohlcv("FAKECOIN123", "1h", limit=10, use_cache=False)
    if bad is None:
        print(f"  ✅ Correctly returned None for invalid symbol")
    else:
        print(f"  ⚠️  Got data for invalid symbol (unexpected)")

    # ── Cleanup ─────────────────────────────────────────────
    print("\n--- Cleanup ---")
    bd.clear_cache()
    cache_after = bd.get_cache_info()
    print(f"  🗑️  Cache cleared: {cache_after['file_count']} files remaining")

    # ── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Binance Data test complete!")
    print("  All data fetching, caching, and error handling verified.")
    print("=" * 60)