import pandas as pd

from data.okx_history import derive_timeframe


def test_incomplete_higher_timeframe_bucket_is_not_emitted():
    index = pd.date_range("2026-01-01", periods=11, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "quote_volume": 100.0,
            "contracts": 1.0,
            "confirmed": True,
        },
        index=index,
    )
    assert derive_timeframe(frame, "1h").empty


def test_completed_hour_is_timestamped_at_bucket_open_for_later_shift():
    index = pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
            "quote_volume": 100.0,
            "contracts": 1.0,
            "confirmed": True,
        },
        index=index,
    )
    result = derive_timeframe(frame, "1h")
    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2026-01-01T00:00:00Z")
