"""Collect current completed OKX swap candles and market snapshots."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from core.market_store import MarketStore
from providers.okx import OKXMarketData

DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
DEFAULT_TIMEFRAMES = ("5m", "15m", "1h")


def instrument_row(symbol: str, raw: dict) -> dict:
    return {
        "instrument_id": raw["instId"],
        "symbol": symbol,
        "asset_type": "swap",
        "settle_currency": raw.get("settleCcy"),
        "contract_value": float(raw["ctVal"]) if raw.get("ctVal") else None,
        "lot_size": float(raw["lotSz"]) if raw.get("lotSz") else None,
        "tick_size": float(raw["tickSz"]) if raw.get("tickSz") else None,
        "state": raw.get("state"),
    }


def collect(
    store: MarketStore,
    provider: OKXMarketData,
    symbols: tuple[str, ...],
    timeframes: tuple[str, ...],
    limit: int,
) -> dict:
    summary = {"symbols": {}, "errors": []}

    for symbol in symbols:
        try:
            raw_instrument = provider.get_instrument(symbol)
            if not raw_instrument:
                raise RuntimeError(f"Instrument not found: {symbol}")
            row = instrument_row(symbol, raw_instrument)
            store.upsert_instrument("okx", row)
            instrument_id = row["instrument_id"]

            candle_counts = {}
            for timeframe in timeframes:
                frame = provider.get_ohlcv(
                    symbol,
                    timeframe,
                    limit=limit,
                    completed_only=True,
                )
                candle_counts[timeframe] = store.upsert_candles(
                    "okx", instrument_id, timeframe, frame
                )

            funding = provider.get_funding_rate(symbol)
            if funding:
                store.upsert_funding(
                    "okx",
                    instrument_id,
                    {
                        "funding_time_ms": int(
                            funding["funding_time"].timestamp() * 1000
                        ),
                        "funding_rate": funding["current_rate"],
                        "premium": funding.get("premium"),
                        "observed_at_ms": int(
                            funding["fetched_at"].timestamp() * 1000
                        ),
                    },
                )

            oi = provider.get_open_interest(symbol)
            if oi:
                store.upsert_open_interest(
                    "okx",
                    instrument_id,
                    {
                        "ts_ms": int(oi["timestamp"].timestamp() * 1000),
                        "oi_contracts": oi["open_interest_contracts"],
                        "oi_base": oi.get("open_interest_base"),
                        "oi_usd": oi.get("open_interest_usd"),
                    },
                )

            summary["symbols"][symbol] = {
                "instrument_id": instrument_id,
                "candles": candle_counts,
                "funding": funding is not None,
                "open_interest": oi is not None,
            }
            print(f"{symbol}: {summary['symbols'][symbol]}")
        except Exception as error:
            message = f"{symbol}: {error}"
            summary["errors"].append(message)
            print("ERROR", message)

    summary["counts"] = store.table_counts()
    summary["collected_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Collect OKX paper market data")
    parser.add_argument("--db", default=os.getenv("MARKET_DB_PATH", "market_data.db"))
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", nargs="*", default=list(DEFAULT_TIMEFRAMES))
    parser.add_argument("--limit", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    store = MarketStore(Path(args.db))
    provider = OKXMarketData()
    if not provider.test_connection():
        raise SystemExit("OKX provider is unavailable")

    summary = collect(
        store,
        provider,
        tuple(args.symbols),
        tuple(args.timeframes),
        args.limit,
    )
    print("counts:", summary["counts"])
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
