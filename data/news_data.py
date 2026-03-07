"""
data/news_data.py
=================
News and social data collector for crypto sentiment analysis.

Sources (all FREE, no API keys):
    - Google News RSS (crypto headlines)
    - CoinDesk RSS
    - CoinTelegraph RSS
    - Reddit JSON (r/cryptocurrency, r/bitcoin)
    - Fear & Greed Index API (alternative.me)

Usage:
    from data.news_data import NewsData
    nd = NewsData()
    all_news = nd.get_all_news(symbol="BTCUSDT")
    headlines = all_news["headlines"]      # List[Dict]
    fg = all_news["fear_greed"]            # Dict or None
"""

import re
import time
import pickle
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from email.utils import parsedate_to_datetime

# ── Import config ────────────────────────────────────────────
try:
    from config import SENTIMENT_CONFIG, CACHE_DIR
except ImportError:
    CACHE_DIR = Path(__file__).parent.parent / "cache"
    SENTIMENT_CONFIG = {
        "max_headlines": 30,
        "sources": {
            "google_news_rss": True,
            "coindesk_rss": True,
            "cointelegraph_rss": True,
            "reddit_cryptocurrency": True,
            "reddit_bitcoin": True,
            "fear_greed_index": True,
        },
    }

from core.logger import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────

RSS_URLS = {
    "google_news": "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
}

REDDIT_URL = "https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1&format=json"

# Map trading pair symbols to search keywords
SYMBOL_KEYWORDS = {
    "BTCUSDT": "Bitcoin BTC crypto",
    "ETHUSDT": "Ethereum ETH crypto",
    "SOLUSDT": "Solana SOL crypto",
    "BNBUSDT": "BNB Binance crypto",
    "XRPUSDT": "XRP Ripple crypto",
}

# Cache TTL
NEWS_CACHE_TTL = 1800  # 30 minutes


# ╔═══════════════════════════════════════════════════════════╗
# ║  NEWS DATA CLASS                                           ║
# ╚═══════════════════════════════════════════════════════════╝

class NewsData:
    """Collects news headlines and social data for sentiment analysis."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/xml, application/rss+xml, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sources = SENTIMENT_CONFIG.get("sources", {})
        self.max_headlines = SENTIMENT_CONFIG.get("max_headlines", 30)
        logger.info("NewsData initialized")

    # ──────────────────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────

    def _fetch_url(self, url: str, timeout: int = 15,
                   max_retries: int = 2) -> Optional[requests.Response]:
        """Fetch a URL with retries and error handling."""
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, timeout=timeout)

                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 30))
                    logger.warning(f"Rate limited on {url[:60]}. Wait {wait}s")
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    logger.warning(
                        f"Server error {resp.status_code} on {url[:60]} "
                        f"(attempt {attempt})"
                    )
                    time.sleep(2 ** attempt)
                    continue

                if resp.status_code == 200:
                    return resp

                logger.warning(f"HTTP {resp.status_code} from {url[:60]}")
                return None

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Timeout: {url[:60]} (attempt {attempt})"
                )
                time.sleep(2)

            except requests.exceptions.ConnectionError:
                logger.warning(
                    f"Connection error: {url[:60]} (attempt {attempt})"
                )
                time.sleep(2)

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                return None

        logger.error(f"All attempts failed: {url[:60]}")
        return None

    def _parse_rss_date(self, date_str: str) -> str:
        """Parse RSS date string to ISO format. Handles multiple formats."""
        if not date_str:
            return datetime.now().isoformat()

        # Try RFC 2822 (standard RSS)
        try:
            dt = parsedate_to_datetime(date_str)
            return dt.isoformat()
        except Exception:
            pass

        # Try ISO format
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.isoformat()
        except Exception:
            pass

        # Try common formats
        for fmt in [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%d %b %Y %H:%M:%S",
        ]:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.isoformat()
            except Exception:
                continue

        return datetime.now().isoformat()

    def _clean_title(self, title: str) -> str:
        """Clean HTML tags and excess whitespace from a headline."""
        if not title:
            return ""
        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", "", title)
        # Remove CDATA markers
        clean = clean.replace("<![CDATA[", "").replace("]]>", "")
        # Normalize whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def _cache_path(self, name: str) -> Path:
        """Generate cache file path."""
        return self.cache_dir / f"news_{name}.pkl"

    def _save_cache(self, path: Path, data):
        """Save data to cache."""
        try:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "data": data,
            }
            with open(path, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            logger.warning(f"News cache save failed: {e}")

    def _load_cache(self, path: Path,
                    max_age: int = NEWS_CACHE_TTL):
        """Load cache if fresh enough."""
        try:
            if not path.exists():
                return None
            with open(path, "rb") as f:
                payload = pickle.load(f)
            cached_time = datetime.fromisoformat(payload["timestamp"])
            age = (datetime.now() - cached_time).total_seconds()
            if age > max_age:
                return None
            logger.debug(f"News cache hit: {path.name} (age: {age:.0f}s)")
            return payload["data"]
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════
    #  GOOGLE NEWS RSS
    # ══════════════════════════════════════════════════════════

    def get_google_news(self, query: str = "cryptocurrency",
                        max_items: int = 10) -> List[Dict]:
        """
        Fetch crypto headlines from Google News RSS.

        Args:
            query:     Search query (e.g., "Bitcoin BTC crypto")
            max_items: Maximum headlines to return

        Returns:
            List of {title, source, url, published, fetched_at}
        """
        if not self.sources.get("google_news_rss", True):
            return []

        url = RSS_URLS["google_news"].format(
            query=requests.utils.quote(query)
        )
        resp = self._fetch_url(url)
        if not resp:
            logger.warning("Google News RSS fetch failed")
            return []

        headlines = []
        try:
            root = ET.fromstring(resp.content)
            channel = root.find("channel")
            if channel is None:
                logger.warning("Google News: no channel in RSS")
                return []

            items = channel.findall("item")
            for item in items[:max_items]:
                title_el = item.find("title")
                link_el = item.find("link")
                pub_el = item.find("pubDate")
                source_el = item.find("source")

                title = self._clean_title(
                    title_el.text if title_el is not None else ""
                )
                if not title:
                    continue

                headlines.append({
                    "title": title,
                    "source": (
                        source_el.text if source_el is not None
                        else "Google News"
                    ),
                    "url": link_el.text if link_el is not None else "",
                    "published": self._parse_rss_date(
                        pub_el.text if pub_el is not None else ""
                    ),
                    "fetched_at": datetime.now().isoformat(),
                    "origin": "google_news",
                })

        except ET.ParseError as e:
            logger.error(f"Google News XML parse error: {e}")
        except Exception as e:
            logger.error(f"Google News processing error: {e}")

        logger.info(f"Google News: {len(headlines)} headlines for '{query}'")
        return headlines

    # ══════════════════════════════════════════════════════════
    #  COINDESK RSS
    # ══════════════════════════════════════════════════════════

    def get_coindesk_news(self, max_items: int = 10) -> List[Dict]:
        """Fetch headlines from CoinDesk RSS feed."""
        if not self.sources.get("coindesk_rss", True):
            return []

        resp = self._fetch_url(RSS_URLS["coindesk"])
        if not resp:
            logger.warning("CoinDesk RSS fetch failed")
            return []

        headlines = []
        try:
            root = ET.fromstring(resp.content)
            channel = root.find("channel")
            if channel is None:
                # Try Atom format
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall("atom:entry", ns)
                for entry in entries[:max_items]:
                    title_el = entry.find("atom:title", ns)
                    link_el = entry.find("atom:link", ns)
                    pub_el = (
                        entry.find("atom:published", ns)
                        or entry.find("atom:updated", ns)
                    )
                    title = self._clean_title(
                        title_el.text if title_el is not None else ""
                    )
                    if not title:
                        continue
                    headlines.append({
                        "title": title,
                        "source": "CoinDesk",
                        "url": (
                            link_el.get("href", "")
                            if link_el is not None else ""
                        ),
                        "published": self._parse_rss_date(
                            pub_el.text if pub_el is not None else ""
                        ),
                        "fetched_at": datetime.now().isoformat(),
                        "origin": "coindesk",
                    })
                logger.info(f"CoinDesk (Atom): {len(headlines)} headlines")
                return headlines

            items = channel.findall("item")
            for item in items[:max_items]:
                title_el = item.find("title")
                link_el = item.find("link")
                pub_el = item.find("pubDate")

                title = self._clean_title(
                    title_el.text if title_el is not None else ""
                )
                if not title:
                    continue

                headlines.append({
                    "title": title,
                    "source": "CoinDesk",
                    "url": link_el.text if link_el is not None else "",
                    "published": self._parse_rss_date(
                        pub_el.text if pub_el is not None else ""
                    ),
                    "fetched_at": datetime.now().isoformat(),
                    "origin": "coindesk",
                })

        except ET.ParseError as e:
            logger.error(f"CoinDesk XML parse error: {e}")
        except Exception as e:
            logger.error(f"CoinDesk processing error: {e}")

        logger.info(f"CoinDesk: {len(headlines)} headlines")
        return headlines

    # ══════════════════════════════════════════════════════════
    #  COINTELEGRAPH RSS
    # ══════════════════════════════════════════════════════════

    def get_cointelegraph_news(self, max_items: int = 10) -> List[Dict]:
        """Fetch headlines from CoinTelegraph RSS feed."""
        if not self.sources.get("cointelegraph_rss", True):
            return []

        resp = self._fetch_url(RSS_URLS["cointelegraph"])
        if not resp:
            logger.warning("CoinTelegraph RSS fetch failed")
            return []

        headlines = []
        try:
            root = ET.fromstring(resp.content)
            channel = root.find("channel")
            if channel is None:
                logger.warning("CoinTelegraph: no channel in RSS")
                return []

            items = channel.findall("item")
            for item in items[:max_items]:
                title_el = item.find("title")
                link_el = item.find("link")
                pub_el = item.find("pubDate")

                title = self._clean_title(
                    title_el.text if title_el is not None else ""
                )
                if not title:
                    continue

                headlines.append({
                    "title": title,
                    "source": "CoinTelegraph",
                    "url": link_el.text if link_el is not None else "",
                    "published": self._parse_rss_date(
                        pub_el.text if pub_el is not None else ""
                    ),
                    "fetched_at": datetime.now().isoformat(),
                    "origin": "cointelegraph",
                })

        except ET.ParseError as e:
            logger.error(f"CoinTelegraph XML parse error: {e}")
        except Exception as e:
            logger.error(f"CoinTelegraph processing error: {e}")

        logger.info(f"CoinTelegraph: {len(headlines)} headlines")
        return headlines

    # ══════════════════════════════════════════════════════════
    #  REDDIT
    # ══════════════════════════════════════════════════════════

    def get_reddit_posts(self, subreddit: str = "cryptocurrency",
                         max_items: int = 10) -> List[Dict]:
        """
        Fetch hot posts from a subreddit via JSON API (no auth needed).

        Args:
            subreddit: e.g., "cryptocurrency", "bitcoin"
            max_items: Maximum posts to return

        Returns:
            List of {title, source, url, score, num_comments, published, fetched_at}
        """
        config_key = f"reddit_{subreddit}"
        if not self.sources.get(config_key, True):
            return []

        url = REDDIT_URL.format(subreddit=subreddit, limit=max_items + 5)
        resp = self._fetch_url(url, timeout=10)

        if not resp:
            logger.warning(f"Reddit r/{subreddit} fetch failed")
            return []

        posts = []
        try:
            data = resp.json()
            children = data.get("data", {}).get("children", [])

            for child in children[:max_items + 5]:
                post = child.get("data", {})

                # Skip pinned / stickied posts
                if post.get("stickied", False):
                    continue

                title = self._clean_title(post.get("title", ""))
                if not title:
                    continue

                # Convert Unix timestamp
                created_utc = post.get("created_utc", 0)
                pub_time = datetime.fromtimestamp(
                    created_utc
                ).isoformat() if created_utc else datetime.now().isoformat()

                posts.append({
                    "title": title,
                    "source": f"r/{subreddit}",
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "score": int(post.get("score", 0)),
                    "num_comments": int(post.get("num_comments", 0)),
                    "published": pub_time,
                    "fetched_at": datetime.now().isoformat(),
                    "origin": f"reddit_{subreddit}",
                })

                if len(posts) >= max_items:
                    break

        except ValueError:
            logger.error(f"Reddit r/{subreddit}: invalid JSON response")
        except Exception as e:
            logger.error(f"Reddit r/{subreddit} processing error: {e}")

        logger.info(f"Reddit r/{subreddit}: {len(posts)} posts")
        return posts

    # ══════════════════════════════════════════════════════════
    #  FEAR & GREED INDEX
    # ══════════════════════════════════════════════════════════

    def get_fear_greed_index(self) -> Optional[Dict]:
        """
        Fetch the Crypto Fear & Greed Index from alternative.me.

        Returns:
            {
                "value": int (0-100),
                "label": str ("Extreme Fear"/"Fear"/"Neutral"/"Greed"/"Extreme Greed"),
                "timestamp": str (ISO format),
            }
            None if fetch fails.
        """
        if not self.sources.get("fear_greed_index", True):
            return None

        resp = self._fetch_url(FEAR_GREED_URL, timeout=10)
        if not resp:
            logger.warning("Fear & Greed API fetch failed")
            return None

        try:
            data = resp.json()
            fng_data = data.get("data", [])
            if not fng_data:
                logger.warning("Fear & Greed: empty data")
                return None

            latest = fng_data[0]
            value = int(latest.get("value", 50))
            label = latest.get("value_classification", "Neutral")

            ts = latest.get("timestamp", "0")
            try:
                ts_dt = datetime.fromtimestamp(int(ts))
                ts_iso = ts_dt.isoformat()
            except (ValueError, OSError):
                ts_iso = datetime.now().isoformat()

            result = {
                "value": value,
                "label": label,
                "timestamp": ts_iso,
            }

            logger.info(f"Fear & Greed Index: {value} ({label})")
            return result

        except ValueError:
            logger.error("Fear & Greed: invalid JSON")
        except Exception as e:
            logger.error(f"Fear & Greed error: {e}")

        return None

    # ══════════════════════════════════════════════════════════
    #  ALL-IN-ONE
    # ══════════════════════════════════════════════════════════

    def get_all_news(self, symbol: str = None,
                     use_cache: bool = True) -> Dict:
        """
        Fetch news from ALL enabled sources.

        Args:
            symbol:    Trading pair for targeted search (e.g., "BTCUSDT")
            use_cache: Use cached results if fresh

        Returns:
            {
                "symbol": str or "GENERAL",
                "fetched_at": str,
                "headlines": List[Dict],   # All headlines combined
                "fear_greed": Dict or None,
                "source_counts": Dict,     # Headlines per source
                "total_headlines": int,
            }
        """
        cache_key = symbol or "GENERAL"

        # Try cache
        if use_cache:
            cache_file = self._cache_path(cache_key)
            cached = self._load_cache(cache_file, NEWS_CACHE_TTL)
            if cached is not None:
                logger.info(
                    f"News cache hit: {cache_key} "
                    f"({cached.get('total_headlines', 0)} headlines)"
                )
                return cached

        logger.info(f"{'─'*50}")
        logger.info(f"Fetching ALL news for {cache_key}...")

        all_headlines = []
        source_counts = {}

        # Determine search query
        query = SYMBOL_KEYWORDS.get(symbol, "cryptocurrency Bitcoin crypto")

        # 1. Google News
        google = self.get_google_news(query=query, max_items=10)
        all_headlines.extend(google)
        source_counts["google_news"] = len(google)
        time.sleep(0.5)

        # 2. CoinDesk
        coindesk = self.get_coindesk_news(max_items=8)
        all_headlines.extend(coindesk)
        source_counts["coindesk"] = len(coindesk)
        time.sleep(0.5)

        # 3. CoinTelegraph
        ct = self.get_cointelegraph_news(max_items=8)
        all_headlines.extend(ct)
        source_counts["cointelegraph"] = len(ct)
        time.sleep(0.5)

        # 4. Reddit r/cryptocurrency
        reddit_cc = self.get_reddit_posts("cryptocurrency", max_items=8)
        all_headlines.extend(reddit_cc)
        source_counts["reddit_cryptocurrency"] = len(reddit_cc)
        time.sleep(0.5)

        # 5. Reddit r/bitcoin
        reddit_btc = self.get_reddit_posts("bitcoin", max_items=6)
        all_headlines.extend(reddit_btc)
        source_counts["reddit_bitcoin"] = len(reddit_btc)

        # 6. Fear & Greed
        fear_greed = self.get_fear_greed_index()

        # Deduplicate by title (case-insensitive)
        seen_titles = set()
        unique_headlines = []
        for h in all_headlines:
            title_lower = h["title"].lower().strip()
            if title_lower not in seen_titles and len(title_lower) > 10:
                seen_titles.add(title_lower)
                unique_headlines.append(h)

        # Sort by published date (newest first)
        try:
            unique_headlines.sort(
                key=lambda x: x.get("published", ""),
                reverse=True,
            )
        except Exception:
            pass  # Keep original order if sort fails

        # Trim to max headlines
        unique_headlines = unique_headlines[:self.max_headlines]

        result = {
            "symbol": cache_key,
            "fetched_at": datetime.now().isoformat(),
            "headlines": unique_headlines,
            "fear_greed": fear_greed,
            "source_counts": source_counts,
            "total_headlines": len(unique_headlines),
        }

        # Cache the result
        if use_cache:
            self._save_cache(self._cache_path(cache_key), result)

        # Summary log
        fg_str = (
            f"{fear_greed['value']} ({fear_greed['label']})"
            if fear_greed else "N/A"
        )
        logger.info(
            f"News complete: {cache_key} | "
            f"{len(unique_headlines)} headlines | "
            f"F&G: {fg_str} | "
            f"Sources: {source_counts}"
        )

        return result

    # ══════════════════════════════════════════════════════════
    #  UTILITIES
    # ══════════════════════════════════════════════════════════

    def clear_cache(self):
        """Clear all news cache files."""
        count = 0
        for f in self.cache_dir.glob("news_*.pkl"):
            f.unlink()
            count += 1
        logger.info(f"News cache cleared: {count} files")


# ╔═══════════════════════════════════════════════════════════╗
# ║  STANDALONE TEST                                           ║
# ╚═══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 60)
    print("  NEWS DATA TEST")
    print("=" * 60)

    nd = NewsData()

    # ── Test 1: Google News ─────────────────────────────────
    print("\n--- Test 1: Google News ---")
    google = nd.get_google_news(query="Bitcoin BTC crypto", max_items=5)
    if google:
        print(f"  ✅ Got {len(google)} headlines")
        for i, h in enumerate(google[:3], 1):
            print(f"     {i}. [{h['source'][:20]}] {h['title'][:70]}...")
    else:
        print("  ⚠️  No Google News headlines (may be blocked in region)")

    # ── Test 2: CoinDesk ────────────────────────────────────
    print("\n--- Test 2: CoinDesk ---")
    coindesk = nd.get_coindesk_news(max_items=5)
    if coindesk:
        print(f"  ✅ Got {len(coindesk)} headlines")
        for i, h in enumerate(coindesk[:3], 1):
            print(f"     {i}. {h['title'][:70]}...")
    else:
        print("  ⚠️  No CoinDesk headlines (feed may have changed)")

    # ── Test 3: CoinTelegraph ───────────────────────────────
    print("\n--- Test 3: CoinTelegraph ---")
    ct = nd.get_cointelegraph_news(max_items=5)
    if ct:
        print(f"  ✅ Got {len(ct)} headlines")
        for i, h in enumerate(ct[:3], 1):
            print(f"     {i}. {h['title'][:70]}...")
    else:
        print("  ⚠️  No CoinTelegraph headlines (feed may have changed)")

    # ── Test 4: Reddit ──────────────────────────────────────
    print("\n--- Test 4: Reddit r/cryptocurrency ---")
    reddit_cc = nd.get_reddit_posts("cryptocurrency", max_items=5)
    if reddit_cc:
        print(f"  ✅ Got {len(reddit_cc)} posts")
        for i, h in enumerate(reddit_cc[:3], 1):
            print(
                f"     {i}. [⬆{h.get('score', 0):>5}] "
                f"{h['title'][:60]}..."
            )
    else:
        print("  ⚠️  No Reddit posts (may be rate limited)")

    print("\n--- Test 4b: Reddit r/bitcoin ---")
    reddit_btc = nd.get_reddit_posts("bitcoin", max_items=5)
    if reddit_btc:
        print(f"  ✅ Got {len(reddit_btc)} posts")
        for i, h in enumerate(reddit_btc[:3], 1):
            print(
                f"     {i}. [⬆{h.get('score', 0):>5}] "
                f"{h['title'][:60]}..."
            )
    else:
        print("  ⚠️  No Reddit posts")

    # ── Test 5: Fear & Greed ────────────────────────────────
    print("\n--- Test 5: Fear & Greed Index ---")
    fg = nd.get_fear_greed_index()
    if fg:
        print(f"  ✅ Value: {fg['value']} ({fg['label']})")
        print(f"  🕐 Time:  {fg['timestamp']}")

        # Visual bar
        bar_len = 30
        filled = int(fg["value"] / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  📊 [{bar}] {fg['value']}/100")
    else:
        print("  ⚠️  Fear & Greed API unavailable")

    # ── Test 6: get_all_news ────────────────────────────────
    print("\n--- Test 6: All News (BTCUSDT) ---")
    all_news = nd.get_all_news(symbol="BTCUSDT", use_cache=False)
    print(f"  📰 Total headlines: {all_news['total_headlines']}")
    print(f"  📊 Sources: {all_news['source_counts']}")
    if all_news["fear_greed"]:
        print(
            f"  😱 Fear & Greed: "
            f"{all_news['fear_greed']['value']} "
            f"({all_news['fear_greed']['label']})"
        )

    print(f"\n  Top 5 headlines:")
    for i, h in enumerate(all_news["headlines"][:5], 1):
        origin = h.get("origin", "?")[:15]
        print(f"     {i}. [{origin:>15}] {h['title'][:60]}")

    # ── Test 7: Cache test ──────────────────────────────────
    print("\n--- Test 7: Cache ---")
    start = time.time()
    cached = nd.get_all_news(symbol="BTCUSDT", use_cache=True)
    elapsed = time.time() - start
    print(f"  ✅ Cached fetch: {elapsed:.3f}s (should be <0.1s)")
    print(f"  ✅ Headlines: {cached['total_headlines']}")

    # ── Test 8: Headline structure validation ───────────────
    print("\n--- Test 8: Data Validation ---")
    required_keys = {"title", "source", "published", "fetched_at", "origin"}
    if all_news["headlines"]:
        sample = all_news["headlines"][0]
        present_keys = set(sample.keys())
        missing = required_keys - present_keys
        if missing:
            print(f"  ❌ Missing keys: {missing}")
        else:
            print(f"  ✅ All required keys present in headline")
        print(f"  ✅ Sample: {sample}")
    else:
        print("  ⚠️  No headlines to validate")

    # Check no empty titles
    empty_titles = [
        h for h in all_news["headlines"] if not h.get("title", "").strip()
    ]
    if empty_titles:
        print(f"  ❌ {len(empty_titles)} empty titles found")
    else:
        print(f"  ✅ No empty titles")

    # ── Test 9: Different symbol ────────────────────────────
    print("\n--- Test 9: All News (ETHUSDT) ---")
    eth_news = nd.get_all_news(symbol="ETHUSDT", use_cache=False)
    print(f"  📰 ETH headlines: {eth_news['total_headlines']}")
    if eth_news["headlines"]:
        print(f"     First: {eth_news['headlines'][0]['title'][:70]}")

    # ── Cleanup ─────────────────────────────────────────────
    print("\n--- Cleanup ---")
    nd.clear_cache()
    print("  🗑️  News cache cleared")

    # ── Summary ─────────────────────────────────────────────
    total_sources_working = sum(
        1 for v in all_news["source_counts"].values() if v > 0
    )
    total_sources = len(all_news["source_counts"])

    print("\n" + "=" * 60)
    print(f"  ✅ News Data test complete!")
    print(f"  📡 Sources working: {total_sources_working}/{total_sources}")
    print(f"  📰 Total headlines: {all_news['total_headlines']}")
    fg_str = (
        f"{all_news['fear_greed']['value']} ({all_news['fear_greed']['label']})"
        if all_news["fear_greed"] else "N/A"
    )
    print(f"  😱 Fear & Greed: {fg_str}")
    print("=" * 60)