"""OKX USDT linear-swap public market-data adapter.

The adapter intentionally provides no authenticated/private/order methods.
Only completed candles are returned by default. OKX returns newest-first;
this adapter normalizes frames to oldest-first UTC DatetimeIndex.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

try:
    from .base import MarketDataProvider
except ImportError:  # direct-file smoke test
    from providers.base import MarketDataProvider


class OKXMarketData(MarketDataProvider):
    BASE_URL = "https://www.okx.com"
    BAR_MAP = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "1d": "1Dutc",
    }

    def __init__(self, timeout: int = 20, retries: int = 3):
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "crypto-agent-v4-paper/1.0",
            }
        )

    @staticmethod
    def to_instrument_id(symbol: str) -> str:
        normalized = symbol.upper().replace("/", "").replace("-", "")
        if normalized.endswith("USDT"):
            base = normalized[:-4]
            return f"{base}-USDT-SWAP"
        raise ValueError(f"Only USDT swap symbols are supported: {symbol}")

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        url = f"{self.BASE_URL}{path}"
        last_error: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=params or {},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if str(payload.get("code")) != "0":
                    raise RuntimeError(
                        f"OKX error code={payload.get('code')} "
                        f"msg={payload.get('msg')}"
                    )
                return payload
            except (requests.RequestException, ValueError, RuntimeError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(2 ** (attempt - 1))

        raise RuntimeError(f"OKX request failed: {url}: {last_error}")

    def test_connection(self) -> bool:
        try:
            payload = self._get("/api/v5/public/time")
            return bool(payload.get("data"))
        except Exception:
            return False

    def get_instrument(self, symbol: str) -> dict[str, Any] | None:
        instrument_id = self.to_instrument_id(symbol)
        payload = self._get(
            "/api/v5/public/instruments",
            {"instType": "SWAP", "instId": instrument_id},
        )
        rows = payload.get("data", [])
        return rows[0] if rows else None

    def get_ohlcv(
        self,
        symbol: str,
        interval: str,
        limit: int = 300,
        completed_only: bool = True,
    ) -> pd.DataFrame:
        if interval not in self.BAR_MAP:
            raise ValueError(
                f"Unsupported interval {interval}; "
                f"supported={sorted(self.BAR_MAP)}"
            )

        limit = max(1, min(int(limit), 300))
        instrument_id = self.to_instrument_id(symbol)
        payload = self._get(
            "/api/v5/market/candles",
            {
                "instId": instrument_id,
                "bar": self.BAR_MAP[interval],
                "limit": str(limit),
            },
        )
        rows = payload.get("data", [])
        columns = [
            "timestamp_ms",
            "open",
            "high",
            "low",
            "close",
            "volume_contracts",
            "volume_base",
            "volume_quote",
            "confirm",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "open", "high", "low", "close", "volume",
                    "quote_volume", "contracts", "confirmed",
                ]
            )

        frame["timestamp_ms"] = pd.to_numeric(
            frame["timestamp_ms"], errors="coerce"
        )
        for column in [
            "open", "high", "low", "close",
            "volume_contracts", "volume_base", "volume_quote",
        ]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame["confirmed"] = frame["confirm"].astype(str).eq("1")
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp_ms"], unit="ms", utc=True
        )
        frame = frame.dropna(
            subset=["timestamp", "open", "high", "low", "close"]
        )
        if completed_only:
            frame = frame[frame["confirmed"]]

        frame = frame.rename(
            columns={
                "volume_base": "volume",
                "volume_quote": "quote_volume",
                "volume_contracts": "contracts",
            }
        )
        frame = (
            frame.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .set_index("timestamp")
        )
        return frame[
            [
                "open", "high", "low", "close", "volume",
                "quote_volume", "contracts", "confirmed",
            ]
        ]

    def get_multi_tf_ohlcv(
        self,
        symbol: str,
        timeframes: tuple[str, ...] = ("5m", "15m", "1h"),
        limit: int = 300,
        completed_only: bool = True,
    ) -> dict[str, pd.DataFrame]:
        return {
            timeframe: self.get_ohlcv(
                symbol,
                timeframe,
                limit=limit,
                completed_only=completed_only,
            )
            for timeframe in timeframes
        }

    def get_ticker(self, symbol: str) -> dict[str, Any] | None:
        instrument_id = self.to_instrument_id(symbol)
        payload = self._get(
            "/api/v5/market/ticker",
            {"instId": instrument_id},
        )
        rows = payload.get("data", [])
        if not rows:
            return None
        row = rows[0]
        return {
            "symbol": symbol,
            "instrument_id": instrument_id,
            "last_price": float(row["last"]),
            "bid_price": float(row["bidPx"]) if row.get("bidPx") else None,
            "ask_price": float(row["askPx"]) if row.get("askPx") else None,
            "high_24h": float(row["high24h"]) if row.get("high24h") else None,
            "low_24h": float(row["low24h"]) if row.get("low24h") else None,
            "volume_24h": float(row["volCcy24h"]) if row.get("volCcy24h") else None,
            "quote_volume_24h": float(row["vol24h"]) if row.get("vol24h") else None,
            "timestamp": datetime.fromtimestamp(
                int(row["ts"]) / 1000, tz=timezone.utc
            ),
        }

    def get_funding_rate(self, symbol: str) -> dict[str, Any] | None:
        instrument_id = self.to_instrument_id(symbol)
        payload = self._get(
            "/api/v5/public/funding-rate",
            {"instId": instrument_id},
        )
        rows = payload.get("data", [])
        if not rows:
            return None
        row = rows[0]
        return {
            "symbol": symbol,
            "instrument_id": instrument_id,
            "current_rate": float(row["fundingRate"]),
            "funding_time": datetime.fromtimestamp(
                int(row["fundingTime"]) / 1000, tz=timezone.utc
            ),
            "next_funding_time": datetime.fromtimestamp(
                int(row["nextFundingTime"]) / 1000, tz=timezone.utc
            ) if row.get("nextFundingTime") else None,
            "premium": float(row["premium"]) if row.get("premium") else None,
            "fetched_at": datetime.now(timezone.utc),
        }

    def get_open_interest(self, symbol: str) -> dict[str, Any] | None:
        instrument_id = self.to_instrument_id(symbol)
        payload = self._get(
            "/api/v5/public/open-interest",
            {"instType": "SWAP", "instId": instrument_id},
        )
        rows = payload.get("data", [])
        if not rows:
            return None
        row = rows[0]
        return {
            "symbol": symbol,
            "instrument_id": instrument_id,
            "open_interest_contracts": float(row["oi"]),
            "open_interest_base": float(row["oiCcy"]),
            "open_interest_usd": float(row["oiUsd"]),
            "timestamp": datetime.fromtimestamp(
                int(row["ts"]) / 1000, tz=timezone.utc
            ),
        }


if __name__ == "__main__":
    provider = OKXMarketData()
    connected = provider.test_connection()
    print("connection:", connected)
    if not connected:
        print(
            "OKX is unreachable from this machine/network. "
            "Run the integration smoke test on the Oracle collector host."
        )
        raise SystemExit(2)

    print("instrument:", provider.get_instrument("BTCUSDT"))
    for timeframe, frame in provider.get_multi_tf_ohlcv(
        "BTCUSDT", ("5m", "15m", "1h"), limit=5
    ).items():
        print(timeframe, len(frame), frame.tail(2))
    print("ticker:", provider.get_ticker("BTCUSDT"))
    print("funding:", provider.get_funding_rate("BTCUSDT"))
    print("open_interest:", provider.get_open_interest("BTCUSDT"))
