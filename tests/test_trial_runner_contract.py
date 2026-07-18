from strategies_v4.triple_barrier import TripleBarrierStrategy


def test_strategy_metadata_is_auditable():
    metadata = TripleBarrierStrategy().metadata()
    assert metadata["strategy_id"] == "triple_barrier_5m"
    assert metadata["version"] == 1
    assert metadata["policy"]["confidence_threshold"] == 0.50
    assert metadata["policy"]["risk_per_trade_pct"] == 0.25
