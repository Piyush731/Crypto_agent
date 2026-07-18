"""Explicit registry of pre-registered v4 strategy trials."""

from strategies_v4.triple_barrier import TripleBarrierStrategy
from strategies_v4.trend_alignment import TrendAlignmentStrategy
from strategies_v4.donchian_15m import Donchian15mStrategy
from strategies_v4.donchian_1h import Donchian1hStrategy
from strategies_v4.range_reversion_15m import RangeReversion15mStrategy

STRATEGIES = {
    "triple_barrier_5m_v1": TripleBarrierStrategy,
    "trend_alignment_5m_v1": TrendAlignmentStrategy,
    "donchian_15m_v1": Donchian15mStrategy,
    "donchian_1h_v1": Donchian1hStrategy,
    "range_reversion_15m_v1": RangeReversion15mStrategy,
}


def get_strategy(strategy_id: str):
    factory = STRATEGIES.get(strategy_id)
    if factory is None:
        raise KeyError(
            f"Unknown strategy {strategy_id}; available={sorted(STRATEGIES)}"
        )
    return factory()
