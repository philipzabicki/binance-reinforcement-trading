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

from common import save_checkpoint, load_checkpoint
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
    ftol=0.0005, period=5, n_max_gen=100, n_max_evals=1_000_000
)

PROBLEM = KeltnerChannelFitting
ALGORITHM = MixedVariableGA
POP_SIZE = 2048
MAX_ITERATIONS = 25
SEARCH_MODE = 'mix'

RESULTS_DIR = os.path.join(REPORT_DIR, "feature_fits_quick")
ACTIONS_FULLPATH = os.path.join(
    REPORT_DIR, "optimal_actions", "final_combined_actions.csv"
)
CHECKPOINT_FILE = os.path.join(RESULTS_DIR,
                               f"keltner_channel_checkpoint_pop{POP_SIZE}_iters{MAX_ITERATIONS}_{SEARCH_MODE}.pkl")


def pool_initializer(ohlcv_np):
    keltner_channel_initializer(ohlcv_np)


def main(max_iterations=10):
    df = pd.read_csv(ACTIONS_FULLPATH)
    ohlcv_np = df.to_numpy()[:, 1:6].astype(float)
    print(f"Loaded from file actions df: {df.head()}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    checkpoint = load_checkpoint(CHECKPOINT_FILE)
    if checkpoint:
        iteration = checkpoint["iteration"]
        unmatched_segments = checkpoint["unmatched_segments"]
        all_results = checkpoint["all_results"]
        print(
            f"Resuming from iteration {iteration} with {len(unmatched_segments)} unmatched segments"
        )
    else:
        iteration = 0
        unmatched_segments = extract_segments_indices(df["Action"].values)
        all_results = []
        print(f"Initial unmatched segments: {len(unmatched_segments)}")

    with Pool(CPU_CORES_COUNT, initializer=pool_initializer, initargs=(ohlcv_np,)) as pool:
        runner = StarmapParallelization(pool.starmap)
        while unmatched_segments.shape[0]:
            print(
                f"Iteration {iteration}, unmatched segments: {len(unmatched_segments)}"
            )

            problem = PROBLEM(df=df, target_segments=unmatched_segments, mode=SEARCH_MODE, elementwise_runner=runner)

            algorithm = ALGORITHM(
                pop_size=POP_SIZE,
                sampling=MixedVariableSampling(),
                mating=MixedVariableMating(
                    eliminate_duplicates=MixedVariableDuplicateElimination()
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
                true_range=TRANGE(*df.to_numpy()[:, 2:5].T.astype(float)),
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
            matched = [
                seg for seg in unmatched_segments if tuple(seg) in gen_segments_set
            ]
            if matched:
                print(f"Matched segments in iteration {iteration}: {len(matched)}")
                print(f"Best params: {best_params}")
                unmatched_segments = np.array(
                    [
                        seg
                        for seg in unmatched_segments
                        if tuple(seg) not in gen_segments_set
                    ]
                )
                all_results.append(
                    {
                        "iteration": iteration,
                        "params": best_params,
                        "matched": len(matched),
                    }
                )
            else:
                print(
                    f"No matched segments in iteration {iteration}: 0, Increased max iterations by 1"
                )
                max_iterations += 1

            save_checkpoint(
                CHECKPOINT_FILE, iteration + 1, unmatched_segments, all_results
            )
            iteration += 1
            if iteration >= max_iterations:
                print("Max iterations reached.")
                break

    output_file = os.path.join(
        RESULTS_DIR, f"keltner_channel_pop{POP_SIZE}_iters{MAX_ITERATIONS}_mode{SEARCH_MODE}.csv"
    )
    pd.DataFrame(all_results).to_csv(output_file, index=False)
    print("All results saved.")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


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
