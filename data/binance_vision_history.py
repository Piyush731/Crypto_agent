"""Download public Binance USD-M monthly 5m archives into an isolated market DB."""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from core.market_store import MarketStore
from data.okx_history import derive_timeframe, gap_report

PROVIDER = "binance_vision"
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"


def month_starts(start: pd.Timestamp, end: pd.Timestamp):
    current = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    final = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    while current <= final:
        yield current
        current = current + pd.offsets.MonthBegin(1)


def archive_url(symbol: str, month: pd.Timestamp) -> str:
    stamp = month.strftime("%Y-%m")
    return f"{BASE_URL}/{symbol}/5m/{symbol}-5m-{stamp}.zip"


def parse_archive(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(names) != 1:
            raise RuntimeError(f"Expected one CSV in archive, found {names}")
        with archive.open(names[0]) as source:
            raw = pd.read_csv(source, header=None)
    if raw.shape[1] < 8:
        raise RuntimeError(f"Unexpected Binance archive columns: {raw.shape[1]}")
    raw = raw.iloc[:, :12].copy()
    raw.columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count", "taker_base",
        "taker_quote", "ignore",
    ]
    raw["open_time"] = pd.to_numeric(raw["open_time"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "quote_volume"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(subset=["open_time", "open", "high", "low", "close"])
    raw["timestamp"] = pd.to_datetime(raw["open_time"], unit="ms", utc=True)
    raw["contracts"] = raw["volume"]
    raw["confirmed"] = True
    return (
        raw[[
            "timestamp", "open", "high", "low", "close", "volume",
            "quote_volume", "contracts", "confirmed",
        ]]
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )


def download(session, url: str, retries: int = 4) -> bytes | None:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=60)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.content
        except requests.RequestException as error:
            last_error = error
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Download failed after retries: {url}: {last_error}")


def collect_symbol(store, session, symbol, start, end, delay):
    instrument_id = symbol
    store.upsert_instrument(
        PROVIDER,
        {
            "instrument_id": instrument_id,
            "symbol": symbol,
            "asset_type": "linear_perpetual_archive",
            "settle_currency": "USDT",
            "state": "historical_public_archive",
        },
    )
    downloaded = 0
    missing_months = []
    upserted = 0
    for month in month_starts(start, end):
        url = archive_url(symbol, month)
        content = download(session, url)
        if content is None:
            missing_months.append(month.strftime("%Y-%m"))
            print(f"  {symbol} {month:%Y-%m}: archive missing", flush=True)
            continue
        frame = parse_archive(content)
        frame = frame[(frame.index >= start) & (frame.index <= end)]
        upserted += store.upsert_candles(PROVIDER, instrument_id, "5m", frame)
        downloaded += 1
        print(f"  {symbol} {month:%Y-%m}: rows={len(frame)}", flush=True)
        time.sleep(delay)

    base = store.get_candles(
        PROVIDER, instrument_id, "5m",
        start_ms=int(start.timestamp() * 1000),
        end_ms=int(end.timestamp() * 1000),
        completed_only=True,
    )
    hourly = derive_timeframe(base, "1h")
    store.upsert_candles(PROVIDER, instrument_id, "1h", hourly)
    return {
        "downloaded_months": downloaded,
        "missing_months": missing_months,
        "upserted": upserted,
        "base": gap_report(base),
        "hourly_rows": len(hourly),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--request-delay", type=float, default=0.05)
    args = parser.parse_args()
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    if start >= end:
        raise SystemExit("start must precede end")
    store = MarketStore(Path(args.db))
    session = requests.Session()
    session.headers["User-Agent"] = "crypto-agent-cross-venue-research/1.0"
    failures = []
    results = {}
    print(f"Range: {start} through {end}", flush=True)
    for symbol in args.symbols:
        print(f"\nDownloading {symbol}...", flush=True)
        try:
            results[symbol] = collect_symbol(
                store, session, symbol, start, end, args.request_delay
            )
            print(results[symbol], flush=True)
        except Exception as error:
            failures.append((symbol, str(error)))
            print(f"ERROR {symbol}: {error}", flush=True)
    print("\nStore counts:", store.table_counts(), flush=True)
    print("Failures:", failures, flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
