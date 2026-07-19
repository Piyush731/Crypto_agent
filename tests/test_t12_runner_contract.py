from research.run_t12_crossvenue import PROVIDER
from research.run_t11_walkforward import INDEPENDENT_END, RECENT_PERIOD_START


def test_t12_is_crossvenue_and_stays_before_recent_period():
    assert PROVIDER == "binance_vision"
    assert INDEPENDENT_END < RECENT_PERIOD_START
