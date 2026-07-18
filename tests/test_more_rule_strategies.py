from strategies_v4.donchian_1h import Donchian1hStrategy
from strategies_v4.range_reversion_15m import RangeReversion15mStrategy


def test_more_trials_are_versioned():
    assert Donchian1hStrategy().strategy_id == "donchian_1h"
    assert Donchian1hStrategy().version == 1
    assert RangeReversion15mStrategy().strategy_id == "range_reversion_15m"
    assert RangeReversion15mStrategy().version == 1
