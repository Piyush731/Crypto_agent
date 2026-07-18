"""Backfill completed OKX 5m swap candles and derive causal 15m/1h bars."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.market_store import MarketStore
from data.okx_collector import instrument_row
from providers.okx import OKXMarketData

PROVIDER = "okx"
BASE_TIMEFRAME = "5m"
PAGE_LIMIT = 300
EXPECTED_PER_BUCKET = {"15m": 3, "1h": 12}
RESAMPLE_RULE = {"15m": "15min", "1h": "1h"}


def parse_rows(rows: list[list[str]]) -> pd.DataFrame:
    columns = [
        "timestamp_ms", "open", "high", "low", "close",
        "contracts", "volume", "quote_volume", "confirm",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    frame["timestamp_ms"] = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
    for column in [
        "open", "high", "low", "close", "contracts", "volume", "quote_volume"
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["confirmed"] = frame["confirm"].astype(str).eq("1")
    frame["timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    frame = frame[frame["confirmed"]]
    return (
        frame[
            [
                "timestamp", "open", "high", "low", "close",
                "volume", "quote_volume", "contracts", "confirmed",
            ]
        ]
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )


def fetch_history_page(
    provider: OKXMarketData,
    instrument_id: str,
    before_open_ms: int | None,
) -> pd.DataFrame:
    params = {
        "instId": instrument_id,
        "bar": "5m",
        "limit": str(PAGE_LIMIT),
    }
    if before_open_ms is not None:
        # OKX `after` returns records earlier than the supplied timestamp.
        params["after"] = str(before_open_ms)
    payload = provider._get("/api/v5/market/history-candles", params)
    return parse_rows(payload.get("data", []))


def derive_timeframe(base: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe not in RESAMPLE_RULE:
        raise ValueError(f"Unsupported derived timeframe: {timeframe}")
    if base.empty:
        return base.copy()

    expected = EXPECTED_PER_BUCKET[timeframe]
    rule = RESAMPLE_RULE[timeframe]
    source = base.sort_index().copy()
    grouped = source.resample(rule, label="left", closed="left", origin="epoch")

    derived = grouped.agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "quote_volume": "sum",
            "contracts": "sum",
            "confirmed": "all",
        }
    )
    counts = grouped["close"].count()
    derived = derived[counts == expected]
    derived = derived.dropna(subset=["open", "high", "low", "close"])
    derived["confirmed"] = True
    return derived


def gap_report(frame: pd.DataFrame, minutes: int = 5) -> dict:
    if len(frame) < 2:
        return {"rows": len(frame), "gaps": 0, "missing_bars": 0}
    deltas = frame.index.to_series().diff().dropna()
    expected = pd.to_timedelta(
        minutes,
        unit="min",
    )
    gap_deltas = deltas[deltas > expected]
    missing = int(sum(max(int(delta / expected) - 1, 0) for delta in gap_deltas))
    return {
        "rows": len(frame),
        "first": frame.index.min().isoformat(),
        "last": frame.index.max().isoformat(),
        "gaps": len(gap_deltas),
        "missing_bars": missing,
        "max_gap_minutes": (
            float(gap_deltas.max() / pd.to_timedelta( 1,  unit="min",))
            if not gap_deltas.empty
            else 0.0
        ),
    }


def backfill_symbol(
    store: MarketStore,
    provider: OKXMarketData,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    request_delay: float,
) -> dict:
    raw_instrument = provider.get_instrument(symbol)
    if not raw_instrument:
        raise RuntimeError(f"Instrument not found: {symbol}")
    info = instrument_row(symbol, raw_instrument)
    store.upsert_instrument(PROVIDER, info)
    instrument_id = info["instrument_id"]

    start_ms = int(start.timestamp() * 1000)
    cursor_ms = int(end.timestamp() * 1000) + 1
    pages = 0
    fetched_rows = 0

    while cursor_ms > start_ms:
        page = fetch_history_page(provider, instrument_id, cursor_ms)
        if page.empty:
            break
        page = page[(page.index >= start) & (page.index <= end)]
        if not page.empty:
            fetched_rows += store.upsert_candles(
                PROVIDER, instrument_id, BASE_TIMEFRAME, page
            )

        oldest_ms = int(fetch_history_page_timestamp_min(page, cursor_ms))
        if oldest_ms >= cursor_ms:
            break
        cursor_ms = oldest_ms - 1
        pages += 1

        if pages % 50 == 0:
            print(
                f"  {symbol}: pages={pages}, cursor="
                f"{pd.to_datetime(cursor_ms, unit='ms', utc=True)}"
            )
        if cursor_ms <= start_ms:
            break
        time.sleep(request_delay)

    base = store.get_candles(
        PROVIDER,
        instrument_id,
        BASE_TIMEFRAME,
        start_ms=start_ms,
        end_ms=int(end.timestamp() * 1000),
        completed_only=True,
    )
    derived_counts = {}
    for timeframe in ("15m", "1h"):
        derived = derive_timeframe(base, timeframe)
        derived_counts[timeframe] = store.upsert_candles(
            PROVIDER, instrument_id, timeframe, derived
        )

    return {
        "instrument_id": instrument_id,
        "pages": pages,
        "upsert_operations": fetched_rows,
        "base": gap_report(base),
        "derived": derived_counts,
    }


def fetch_history_page_timestamp_min(page: pd.DataFrame, fallback_ms: int) -> int:
    if page.empty:
        # The filtered page can be empty because all returned rows predate start.
        return fallback_ms - PAGE_LIMIT * 5 * 60 * 1000
    return int(page.index.min().timestamp() * 1000)


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill OKX completed 5m history")
    parser.add_argument("--db", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--request-delay", type=float, default=0.12)
    return parser.parse_args()


def main():
    args = parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    end = (
        pd.Timestamp(args.end, tz="UTC")
        if args.end
        else pd.Timestamp.now(tz="UTC").floor("5min")
    )
    if start >= end:
        raise SystemExit("start must be before end")

    store = MarketStore(Path(args.db))
    provider = OKXMarketData(timeout=30, retries=4)
    if not provider.test_connection():
        raise SystemExit("OKX is unavailable")

    print(f"Range: {start} through {end}")
    results = {}
    failures = []
    for symbol in args.symbols:
        print(f"\nBackfilling {symbol}...")
        try:
            results[symbol] = backfill_symbol(
                store, provider, symbol, start, end, args.request_delay
            )
            print(results[symbol])
        except Exception as error:
            failures.append((symbol, str(error)))
            print(f"ERROR {symbol}: {error}")

    print("\nStore counts:", store.table_counts())
    print("Failures:", failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
