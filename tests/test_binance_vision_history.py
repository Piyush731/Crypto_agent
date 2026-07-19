import io
import zipfile

from data.binance_vision_history import archive_url, parse_archive


def test_archive_url_is_public_usdm_monthly_5m():
    import pandas as pd
    url = archive_url("BTCUSDT", pd.Timestamp("2021-01-01T00:00:00Z"))
    assert url == (
        "https://data.binance.vision/data/futures/um/monthly/klines/"
        "BTCUSDT/5m/BTCUSDT-5m-2021-01.zip"
    )


def test_parse_archive_normalizes_completed_ohlcv():
    csv = (
        "1609459200000,100,101,99,100.5,12,1609459499999,1206,10,6,603,0\n"
        "1609459500000,100.5,102,100,101,13,1609459799999,1313,11,7,707,0\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("BTCUSDT-5m-2021-01.csv", csv)
    frame = parse_archive(buffer.getvalue())
    assert len(frame) == 2
    assert frame.index[0].isoformat() == "2021-01-01T00:00:00+00:00"
    assert bool(frame.iloc[0]["confirmed"]) is True
    assert float(frame.iloc[1]["close"]) == 101.0
