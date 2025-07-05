import os
import pickle

import pandas as pd


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
