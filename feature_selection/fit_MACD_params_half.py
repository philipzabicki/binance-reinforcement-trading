import cProfile
import io
import os
import pstats
import time
# import numpy as np
from multiprocessing import Pool

import numpy as np
import pandas as pd
# from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.mixed import MixedVariableGA
# import matplotlib.pyplot as plt
from pymoo.core.mixed import (
    MixedVariableMating,
    MixedVariableSampling,
    MixedVariableDuplicateElimination,
)
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

from common import save_checkpoint, load_checkpoint, build_half_df
from definitions import REPORT_DIR
from genetic_classes.feature_action_fitter import (
    MACDFitting,
)
from utils.feature_generation import custom_MACD
from utils.feature_generation import ohlcv_initializer
from utils.ta_tools import (
    MACD_lines_cross_with_zero,
    MACD_lines_cross,
    MACD_lines_approaching_cross_with_zero,
    MACD_lines_approaching_cross,
    MACD_signal_line_zero_cross,
    MACD_line_zero_cross,
    MACD_histogram_reversal,
)
# from utils.miscellaneous import convert_variables
from utils.ta_tools import extract_segments_indices

CPU_CORES_COUNT = 17
print(f"CPUs used: {CPU_CORES_COUNT}")

TERMINATION = DefaultMultiObjectiveTermination(
    ftol=0.0001, period=5, n_max_gen=100, n_max_evals=1_000_000
)

PROBLEM = MACDFitting
ALGORITHM = MixedVariableGA
POP_SIZE = 8192
MAX_ITERATIONS = 15
SEARCH_MODE = 'mix'

RESULTS_DIR = os.path.join(REPORT_DIR, "feature_fits_quick")
ACTIONS_FULLPATH = os.path.join(
    REPORT_DIR, "optimal_actions", "final_combined_actions.csv"
)
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, f"macd_checkpoint_pop{POP_SIZE}_iters{MAX_ITERATIONS}_{SEARCH_MODE}.pkl")


def pool_initializer(ohlcv_np):
    ohlcv_initializer(ohlcv_np)


def run_half(df_h: pd.DataFrame,
             half_idx: int,
             ohlcv_np: np.ndarray,
             max_iterations: int = 15):
    """
    GA fitting pass for MACD on the selected half of data
    (0 = ≤ median, 1 = > median). Checkpoints/outputs carry suffix _h{half_idx}.
    """
    checkpoint_file = CHECKPOINT_FILE.replace(".pkl", f"_h{half_idx}.pkl")
    output_file = os.path.join(
        RESULTS_DIR,
        f"macd_pop{POP_SIZE}_iters{max_iterations}_mode{SEARCH_MODE}_h{half_idx}.csv"
    )

    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        iteration = checkpoint["iteration"]
        unmatched_segments = checkpoint["unmatched_segments"]
        all_results = checkpoint["all_results"]
        print(f"[h{half_idx}] Resuming at iter {iteration} with {len(unmatched_segments)} segments left")
    else:
        iteration = 0
        unmatched_segments = extract_segments_indices(df_h["Action"].values)
        all_results = []
        print(f"[h{half_idx}] Starting – initial segments: {len(unmatched_segments)}")

    if unmatched_segments.shape[0] == 0:
        print(f"[h{half_idx}] No actionable segments – skip.")
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
                    eliminate_duplicates=MixedVariableDuplicateElimination()
                ),
                eliminate_duplicates=MixedVariableDuplicateElimination(),
            )

            res = minimize(problem,
                           algorithm,
                           save_history=False,
                           termination=TERMINATION,
                           verbose=True)

            best_params = res.X
            macd, macd_signal = custom_MACD(
                ohlcv_np,
                fast_source=best_params["fast_source"],
                slow_source=best_params["slow_source"],
                fast_ma_type=best_params["fast_ma_type"],
                fast_period=best_params["fast_period"],
                slow_ma_type=best_params["slow_ma_type"],
                slow_period=best_params["slow_period"],
                signal_ma_type=best_params["signal_ma_type"],
                signal_period=best_params["signal_period"],
            )

            signal_func_mapping = {
                "lines_cross_with_zero": MACD_lines_cross_with_zero,
                "lines_cross": MACD_lines_cross,
                "lines_approaching_cross_with_zero": MACD_lines_approaching_cross_with_zero,
                "lines_approaching_cross": MACD_lines_approaching_cross,
                "signal_line_zero_cross": MACD_signal_line_zero_cross,
                "MACD_line_zero_cross": MACD_line_zero_cross,
                "histogram_reversal": MACD_histogram_reversal,
            }
            signals = np.array(signal_func_mapping[best_params["signal_type"]](macd, macd_signal))
            gen_segments = extract_segments_indices(signals)
            gen_segments_set = {tuple(seg) for seg in gen_segments}

            matched = sum(1 for seg in unmatched_segments if tuple(seg) in gen_segments_set)
            if matched:
                unmatched_segments = np.array(
                    [seg for seg in unmatched_segments if tuple(seg) not in gen_segments_set]
                )
                all_results.append(
                    {"iteration": iteration, "params": best_params, "matched": matched}
                )
                print(f"[h{half_idx}] matched: {matched}")
            else:
                print(f"[h{half_idx}] no match, extend budget by 1")
                max_iterations += 1  # usuń, jeśli limit 15 ma być sztywny

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
    # 1. Load dataframe once
    df = pd.read_csv(ACTIONS_FULLPATH)
    ohlcv_np = df.to_numpy()[:, 1:6].astype(float)

    # 2. Median split
    median_val = df["Weight"].median()
    print(f"Median Weight: {median_val}")

    # 3. Process two halves
    for h_idx in (0, 1):
        df_h = build_half_df(df, h_idx, median_val)
        active_rows = (df_h["Action"] != 0).sum()
        print(f"\n=== Half {h_idx} – active rows: {active_rows} ===")
        run_half(df_h,
                 half_idx=h_idx,
                 ohlcv_np=ohlcv_np,
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
