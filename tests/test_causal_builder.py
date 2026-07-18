from pathlib import Path

import pandas as pd

from core.market_store import MarketStore
from data.okx_history import derive_timeframe
from features.causal_builder import CausalFeatureBuilder


def base_frame(rows=1200):
    index = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    close = pd.Series(range(rows), index=index, dtype=float) + 100
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 10.0,
            "quote_volume": close * 10,
            "contracts": 100.0,
            "confirmed": True,
        },
        index=index,
    )


def make_store(path: Path):
    store = MarketStore(path)
    store.upsert_instrument(
        "okx",
        {
            "instrument_id": "BTC-USDT-SWAP",
            "symbol": "BTCUSDT",
            "asset_type": "swap",
            "settle_currency": "USDT",
        },
    )
    base = base_frame()
    store.upsert_candles("okx", "BTC-USDT-SWAP", "5m", base)
    store.upsert_candles(
        "okx", "BTC-USDT-SWAP", "15m", derive_timeframe(base, "15m")
    )
    store.upsert_candles(
        "okx", "BTC-USDT-SWAP", "1h", derive_timeframe(base, "1h")
    )
    return store


def test_builder_is_aligned_and_purged_metadata_exists(tmp_path):
    builder = CausalFeatureBuilder(make_store(tmp_path / "market.db"))
    result = builder.build("BTC-USDT-SWAP", include_target=True)
    assert not result["features"].empty
    assert result["features"].index.equals(result["target"].index)
    assert result["features"].index.equals(result["target_end_time"].index)
    assert (result["target_end_time"] > result["features"].index).all()
    assert set(result["target"].unique()).issubset({-1, 0, 1})


def test_future_hour_change_cannot_affect_earlier_decision(tmp_path):
    store = make_store(tmp_path / "market.db")
    builder = CausalFeatureBuilder(store)
    first = builder.build("BTC-USDT-SWAP", include_target=False)["features"]

    one_hour = store.get_candles("okx", "BTC-USDT-SWAP", "1h")
    last_time = one_hour.index[-1]
    one_hour.loc[last_time, "close"] *= 10
    store.upsert_candles("okx", "BTC-USDT-SWAP", "1h", one_hour.loc[[last_time]])

    second = builder.build("BTC-USDT-SWAP", include_target=False)["features"]
    cutoff = last_time + pd.Timedelta(hours=1)
    common = first.index.intersection(second.index)
    earlier = common[common < cutoff]
    pd.testing.assert_frame_equal(first.loc[earlier], second.loc[earlier])
