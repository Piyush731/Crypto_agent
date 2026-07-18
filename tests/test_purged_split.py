import pandas as pd

from models.purged_split import PurgedWalkForwardSplit, final_holdout_split


def make_labels(rows=1000, horizon=48):
    index = pd.date_range("2025-01-01", periods=rows, freq="5min", tz="UTC")
    ends = pd.Series(index, index=index).shift(-horizon)
    valid = ends.notna()
    return index[valid], ends.loc[valid]


def test_folds_have_no_label_overlap():
    index, ends = make_labels()
    splitter = PurgedWalkForwardSplit(
        n_splits=3, min_train_size=300, test_size=150
    )
    folds = list(splitter.split(index, ends))
    assert folds
    for fold in folds:
        assert ends.iloc[fold.train_indices].max() < fold.test_start_time
        assert fold.train_indices.max() < fold.test_indices.min()


def test_final_holdout_is_purged():
    index, ends = make_labels()
    train, holdout = final_holdout_split(index, ends, 0.2)
    assert ends.iloc[train].max() < index[holdout[0]]
    assert train.max() < holdout.min()
