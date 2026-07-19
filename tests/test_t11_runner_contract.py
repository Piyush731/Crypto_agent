from research.run_t11_walkforward import (
    DATA_START,
    EVALUATION_START,
    FINAL_HOLDOUT_START,
    INDEPENDENT_END,
    RECENT_PERIOD_START,
    WINDOWS,
)


def test_t11_date_bounds_do_not_touch_recent_or_final_periods():
    assert DATA_START.isoformat() == "2021-01-01T00:00:00+00:00"
    assert EVALUATION_START.isoformat() == "2021-04-05T00:00:00+00:00"
    assert INDEPENDENT_END < RECENT_PERIOD_START
    assert INDEPENDENT_END < FINAL_HOLDOUT_START
    assert len(WINDOWS) == 4
    assert all(start <= end <= INDEPENDENT_END for _, start, end in WINDOWS)
