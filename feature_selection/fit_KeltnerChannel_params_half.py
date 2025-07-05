import cProfile
import io
import os
import pstats
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.mixed import (
    MixedVariableMating,
    MixedVariableSampling,
    MixedVariableDuplicateElimination,
)
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from talib import TRANGE

from common import save_checkpoint, load_checkpoint, build_half_df
from definitions import REPORT_DIR
from genetic_classes.feature_action_fitter import KeltnerChannelFitting
from utils.feature_generation import (
    custom_keltner_channel_signal,
    keltner_channel_initializer,
)
from utils.ta_tools import extract_segments_indices

CPU_CORES_COUNT = 17
print(f"CPUs used: {CPU_CORES_COUNT}")

TERMINATION = DefaultMultiObjectiveTermination(
    ftol=0.0001, period=5, n_max_gen=100, n_max_evals=1_000_000
)

PROBLEM = KeltnerChannelFitting
ALGORITHM = MixedVariableGA
POP_SIZE = 8192
MAX_ITERATIONS = 15
SEARCH_MODE = 'mix'

RESULTS_DIR = os.path.join(REPORT_DIR, "feature_fits_quick")
ACTIONS_FULLPATH = os.path.join(
    REPORT_DIR, "optimal_actions", "final_combined_actions.csv"
)
CHECKPOINT_FILE = os.path.join(RESULTS_DIR,
                               f"keltner_channel_checkpoint_pop{POP_SIZE}_iters{MAX_ITERATIONS}_{SEARCH_MODE}.pkl")


def pool_initializer(ohlcv_np):
    keltner_channel_initializer(ohlcv_np)


def run_half(df_h: pd.DataFrame,
             half_idx: int,
             ohlcv_np: np.ndarray,
             trange_arr: np.ndarray,
             max_iterations: int = 15):
    """
    GA fitting pass for one half of the data (0 = ≤ median, 1 = > median).
    Saves checkpoint/output files suffixed with *_h{half_idx}.
    """
    checkpoint_file = CHECKPOINT_FILE.replace(".pkl", f"_h{half_idx}.pkl")
    output_file = os.path.join(
        RESULTS_DIR,
        f"keltner_channel_pop{POP_SIZE}_iters{max_iterations}_mode{SEARCH_MODE}_h{half_idx}.csv"
    )

    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        iteration = checkpoint["iteration"]
        unmatched_segments = checkpoint["unmatched_segments"]
        all_results = checkpoint["all_results"]
        print(f"[h{half_idx}] Resuming from iter {iteration} – segments left: {len(unmatched_segments)}")
    else:
        iteration = 0
        unmatched_segments = extract_segments_indices(df_h["Action"].values)
        all_results = []
        print(f"[h{half_idx}] Starting – initial segments: {len(unmatched_segments)}")

    if unmatched_segments.shape[0] == 0:
        print(f"[h{half_idx}] No actionable segments, skipping.")
        return

    with Pool(CPU_CORES_COUNT,
              initializer=pool_initializer,
              initargs=(ohlcv_np,)) as pool:
        runner = StarmapParallelization(pool.starmap)

        while unmatched_segments.shape[0]:
            print(f"[h{half_idx}] iter {iteration}, segments: {len(unmatched_segments)}")

            problem = PROBLEM(df=df_h,
                              target_segments=unmatched_segments,
                              mode=SEARCH_MODE,
                              elementwise_runner=runner)

            algorithm = ALGORITHM(
                pop_size=POP_SIZE,
                sampling=MixedVariableSampling(),
                mating=MixedVariableMating(
                    eliminate_duplicates=MixedVariableDuplicateElimination(),
                ),
                eliminate_duplicates=MixedVariableDuplicateElimination(),
            )

            res = minimize(
                problem,
                algorithm,
                save_history=False,
                termination=TERMINATION,
                verbose=True,
            )

            best_params = res.X
            signals = custom_keltner_channel_signal(
                ohlcv=ohlcv_np,
                true_range=trange_arr,
                ma_type=best_params["ma_type"],
                ma_period=best_params["ma_period"],
                atr_ma_type=best_params["atr_ma_type"],
                atr_period=best_params["atr_period"],
                atr_multi=best_params["atr_multi"],
                source=best_params["source"],
            )
            signals = np.where(signals >= 1, 1, np.where(signals <= -1, -1, 0))
            gen_segments = extract_segments_indices(signals)
            gen_segments_set = {tuple(seg) for seg in gen_segments}

            matched = sum(1 for seg in unmatched_segments if tuple(seg) in gen_segments_set)
            if matched:
                print(f"[h{half_idx}] matched: {matched}")
                unmatched_segments = np.array(
                    [seg for seg in unmatched_segments if tuple(seg) not in gen_segments_set]
                )
                all_results.append(
                    {"iteration": iteration, "params": best_params, "matched": matched}
                )
            else:
                print(f"[h{half_idx}] no match, extend budget by 1")
                max_iterations += 1  # opcjonalnie usuń, jeśli 15 ma być „twardym” limitem

            save_checkpoint(checkpoint_file, iteration + 1, unmatched_segments, all_results)
            iteration += 1
            if iteration >= max_iterations:
                print(f"[h{half_idx}] reached {max_iterations} iterations.")
                break

    pd.DataFrame(all_results).to_csv(output_file, index=False)
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    print(f"[h{half_idx}] finished – results saved to {output_file}")


def main(max_iterations: int = MAX_ITERATIONS):
    # 1. Read once
    df = pd.read_csv(ACTIONS_FULLPATH)
    ohlcv_np = df.to_numpy()[:, 1:6].astype(float)
    trange_arr = TRANGE(*df.to_numpy()[:, 2:5].T.astype(float))  # high-low-close → TRANGE

    # 2. Median split
    median_val = df["Weight"].median()
    print(f"Median Weight: {median_val}")

    # 3. Process both halves
    for h_idx in (0, 1):
        df_h = build_half_df(df, h_idx, median_val)
        active_rows = (df_h["Action"] != 0).sum()
        print(f"\n=== Half {h_idx} – active rows: {active_rows} ===")
        run_half(df_h,
                 half_idx=h_idx,
                 ohlcv_np=ohlcv_np,
                 trange_arr=trange_arr,
                 max_iterations=max_iterations)  # stały limit


def profile_main():
    profiler = cProfile.Profile()
    profiler.enable()
    main(MAX_ITERATIONS)
    profiler.disable()
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats("tottime")
    stats.print_stats()
    print(s.getvalue())


if __name__ == "__main__":
    start_time = time.time()  # start timing
    main(MAX_ITERATIONS)
    end_time = time.time()  # end timing
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    # profile_main()
