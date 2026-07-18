from strategies_v4.registry import STRATEGIES, get_strategy


def test_registry_contains_only_pre_registered_trial():
    assert set(STRATEGIES) == {"triple_barrier_5m_v1"}
    assert get_strategy("triple_barrier_5m_v1").strategy_id == "triple_barrier_5m"
