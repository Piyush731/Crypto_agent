"""Explicit registry of pre-registered v4 strategy trials."""

from strategies_v4.triple_barrier import TripleBarrierStrategy

STRATEGIES = {
    "triple_barrier_5m_v1": TripleBarrierStrategy,
}


def get_strategy(strategy_id: str):
    factory = STRATEGIES.get(strategy_id)
    if factory is None:
        raise KeyError(
            f"Unknown strategy {strategy_id}; available={sorted(STRATEGIES)}"
        )
    return factory()
