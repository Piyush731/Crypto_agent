import pandas as pd
import pytest

from providers.okx import OKXMarketData


def test_symbol_mapping():
    assert OKXMarketData.to_instrument_id("BTCUSDT") == "BTC-USDT-SWAP"
    assert OKXMarketData.to_instrument_id("ETH/USDT") == "ETH-USDT-SWAP"


@pytest.mark.integration
def test_completed_candles_are_sorted_and_confirmed():
    provider = OKXMarketData()
    if not provider.test_connection():
        pytest.skip("OKX is unavailable from this machine/network")

    frame = provider.get_ohlcv("BTCUSDT", "5m", limit=10, completed_only=True)
    assert not frame.empty
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique
    assert frame["confirmed"].all()
    assert (frame[["open", "high", "low", "close"]] > 0).all().all()
