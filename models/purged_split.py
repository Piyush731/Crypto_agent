"""Purged chronological splits for overlapping financial labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedFold:
    fold: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_end_time: pd.Timestamp
    test_start_time: pd.Timestamp


class PurgedWalkForwardSplit:
    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 10_000,
        test_size: int = 5_000,
        embargo_bars: int = 0,
    ):
        if n_splits < 1 or min_train_size < 1 or test_size < 1:
            raise ValueError("split sizes must be positive")
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.embargo_bars = max(0, embargo_bars)

    def split(
        self,
        index: pd.DatetimeIndex,
        target_end_time: pd.Series,
    ) -> Iterator[PurgedFold]:
        if not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError("index must be sorted and unique")
        ends = pd.to_datetime(target_end_time.reindex(index), utc=True)
        n = len(index)
        possible = max((n - self.min_train_size) // self.test_size, 0)
        folds = min(self.n_splits, possible)
        if folds < 1:
            raise ValueError("insufficient rows for requested purged split")

        first_test_start = n - folds * self.test_size
        for fold_number in range(folds):
            test_start = first_test_start + fold_number * self.test_size
            test_end = min(test_start + self.test_size, n)
            test_indices = np.arange(test_start, test_end)
            test_start_time = index[test_start]

            candidate_train = np.arange(0, test_start)
            no_label_overlap = ends.iloc[candidate_train] < test_start_time
            train_indices = candidate_train[no_label_overlap.to_numpy()]
            if self.embargo_bars:
                train_indices = train_indices[: max(len(train_indices) - self.embargo_bars, 0)]
            if len(train_indices) < self.min_train_size:
                continue

            yield PurgedFold(
                fold=fold_number + 1,
                train_indices=train_indices,
                test_indices=test_indices,
                train_end_time=index[train_indices[-1]],
                test_start_time=test_start_time,
            )


def final_holdout_split(
    index: pd.DatetimeIndex,
    target_end_time: pd.Series,
    holdout_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5")
    split_at = int(len(index) * (1 - holdout_fraction))
    holdout = np.arange(split_at, len(index))
    holdout_start = index[split_at]
    ends = pd.to_datetime(target_end_time.reindex(index), utc=True)
    candidates = np.arange(0, split_at)
    train = candidates[(ends.iloc[candidates] < holdout_start).to_numpy()]
    if len(train) == 0 or len(holdout) == 0:
        raise ValueError("empty train or holdout after purge")
    return train, holdout
