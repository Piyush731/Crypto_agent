from models.v4_baselines import MODEL_NAMES


def test_model_trials_are_pre_registered():
    assert MODEL_NAMES == ("dummy", "logistic", "hist_gb")
