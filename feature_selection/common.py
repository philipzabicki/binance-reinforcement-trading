import os
import pickle

import pandas as pd
import numpy as np


def save_checkpoint(file_fullpath, iteration, unmatched_segments, all_results):
    checkpoint = {
        "iteration": iteration,
        "unmatched_segments": unmatched_segments,
        "all_results": all_results,
    }
    with open(file_fullpath, "wb") as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(file_fullpath):
    if os.path.exists(file_fullpath):
        with open(file_fullpath, "rb") as f:
            return pickle.load(f)
    else:
        return None


def build_half_df(df: pd.DataFrame, half_idx: int, median_val: float) -> pd.DataFrame:
    """
    Return a copy of *df* in which column 'Action' is kept only for
    the requested half_idx (0 = ≤ median, 1 = > median).
    Everywhere else 'Action' is set to 0.
    """
    hdf = df.copy()
    if half_idx == 0:
        mask = hdf["Weight"] <= median_val
    else:  # 1
        mask = hdf["Weight"] > median_val

    hdf.loc[~mask, "Action"] = 0
    return hdf

def build_bin_df(df, bin_idx, assignments):
    hdf = df.copy()
    mask = (assignments == bin_idx)
    hdf.loc[~mask, "Action"] = 0
    hdf.loc[~mask, "Weight"] = 1.0
    return hdf

def _equal_freq_assignments(weight_s, n_splits):
    if n_splits < 0:
        raise ValueError("n_splits must be >= 0")
    if n_splits == 0:
        return None

    assert not weight_s.isna().any(), "Weight contains NaN values"

    weights = weight_s.to_numpy()
    active = weights != 1.0
    n_active = int(active.sum())
    if n_active == 0:
        raise ValueError("No rows with Weight != 0")
    if n_splits > n_active:
        raise ValueError("n_splits cannot exceed the number of rows with Weight != 0")

    ranks = weight_s[active].rank(method="average").to_numpy()
    # print(f'ranks {ranks}')
    bins_active = np.floor(ranks * n_splits / n_active).astype(int)
    # print(f'bins_active {bins_active}')
    bins_active = np.clip(bins_active, 0, n_splits - 1)
    # print(f'bins_active {bins_active}')

    assignments = np.full(len(weight_s), -1, dtype=int)
    # print(f'assignments {assignments}')
    assignments[active] = bins_active
    # print(f'assignments {assignments}')
    return assignments