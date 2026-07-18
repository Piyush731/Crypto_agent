from strategies_v4.registry import STRATEGIES, get_strategy


def test_registry_contains_only_pre_registered_trials():
    assert set(STRATEGIES) == {
        "triple_barrier_5m_v1",
        "trend_alignment_5m_v1",
        "donchian_15m_v1",
        "donchian_1h_v1",
        "range_reversion_15m_v1",
    }
    assert get_strategy("triple_barrier_5m_v1").strategy_id == "triple_barrier_5m"
    assert get_strategy("trend_alignment_5m_v1").strategy_id == "trend_alignment_5m"
    assert get_strategy("donchian_15m_v1").strategy_id == "donchian_15m"
    assert get_strategy("donchian_1h_v1").strategy_id == "donchian_1h"
    assert get_strategy("range_reversion_15m_v1").strategy_id == "range_reversion_15m"
