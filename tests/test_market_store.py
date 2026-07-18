from pathlib import Path

import pandas as pd

from core.market_store import MarketStore


def sample_frame(close_value: float = 101.0):
    index = pd.DatetimeIndex(
        [pd.Timestamp("2026-01-01T00:00:00Z")], name="timestamp"
    )
    return pd.DataFrame(
        {
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [close_value],
            "volume": [10.0],
            "quote_volume": [1010.0],
            "contracts": [1000.0],
            "confirmed": [True],
        },
        index=index,
    )


def test_candle_upsert_is_idempotent(tmp_path: Path):
    store = MarketStore(tmp_path / "market.db")
    store.upsert_instrument(
        "okx",
        {
            "instrument_id": "BTC-USDT-SWAP",
            "symbol": "BTCUSDT",
            "asset_type": "swap",
            "settle_currency": "USDT",
        },
    )

    assert store.upsert_candles(
        "okx", "BTC-USDT-SWAP", "5m", sample_frame(101.0)
    ) == 1
    assert store.upsert_candles(
        "okx", "BTC-USDT-SWAP", "5m", sample_frame(103.0)
    ) == 1

    result = store.get_candles("okx", "BTC-USDT-SWAP", "5m")
    assert len(result) == 1
    assert result.iloc[0]["close"] == 103.0
    assert store.table_counts()["candles"] == 1


def test_incomplete_candle_filter(tmp_path: Path):
    store = MarketStore(tmp_path / "market.db")
    store.upsert_instrument(
        "okx",
        {
            "instrument_id": "BTC-USDT-SWAP",
            "symbol": "BTCUSDT",
            "asset_type": "swap",
            "settle_currency": "USDT",
        },
    )
    frame = sample_frame()
    frame["confirmed"] = False
    store.upsert_candles("okx", "BTC-USDT-SWAP", "5m", frame)

    assert store.get_candles(
        "okx", "BTC-USDT-SWAP", "5m", completed_only=True
    ).empty
    assert len(
        store.get_candles(
            "okx", "BTC-USDT-SWAP", "5m", completed_only=False
        )
    ) == 1
