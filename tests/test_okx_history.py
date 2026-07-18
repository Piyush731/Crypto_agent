import pandas as pd

from data.okx_history import derive_timeframe, gap_report


def make_5m(rows: int):
    index = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    values = list(range(rows))
    return pd.DataFrame(
        {
            "open": [100 + value for value in values],
            "high": [101 + value for value in values],
            "low": [99 + value for value in values],
            "close": [100.5 + value for value in values],
            "volume": [1.0] * rows,
            "quote_volume": [100.0] * rows,
            "contracts": [10.0] * rows,
            "confirmed": [True] * rows,
        },
        index=index,
    )


def test_derive_15m_requires_three_complete_rows():
    frame = make_5m(7)
    derived = derive_timeframe(frame, "15m")
    assert len(derived) == 2
    assert derived.iloc[0]["open"] == 100
    assert derived.iloc[0]["close"] == 102.5
    assert derived.iloc[0]["volume"] == 3


def test_derive_1h_requires_twelve_complete_rows():
    frame = make_5m(13)
    derived = derive_timeframe(frame, "1h")
    assert len(derived) == 1
    assert derived.iloc[0]["open"] == 100
    assert derived.iloc[0]["close"] == 111.5


def test_gap_report_counts_missing_bars():
    frame = make_5m(5).drop(make_5m(5).index[2])
    report = gap_report(frame)
    assert report["gaps"] == 1
    assert report["missing_bars"] == 1
