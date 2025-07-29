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
from pymoo.core.variable import Real, Integer
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation as PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

from common import save_checkpoint, load_checkpoint
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

PROBLEM = MACDFitting
ALGORITHM = MixedVariableGA
TERMINATION = DefaultMultiObjectiveTermination(
    # cvtol=1e-8, # default 1e-8
    xtol=0.00005,  # default 0.0005
    ftol=0.00001,  # default 0.005
    period=7,
    n_max_gen=100,
    n_max_evals=1_000_000
)
MATING = MixedVariableMating(
    mutation={Real: PM(eta=10),
              Integer: PM(eta=10, vtype=float, repair=RoundingRepair())
              },
    crossover={Real: SBX(eta=5),
               Integer: SBX(eta=5, vtype=float, repair=RoundingRepair())
               },
    eliminate_duplicates=MixedVariableDuplicateElimination())

CPU_CORES_COUNT = 17
print(f"CPUs used: {CPU_CORES_COUNT}")
POP_SIZE = 2048
MAX_ITERATIONS = 25
SEARCH_MODE = 'mix'

RESULTS_DIR = os.path.join(REPORT_DIR, "feature_fits_quick")
ACTIONS_FULLPATH = os.path.join(
    REPORT_DIR, "optimal_actions", "final_combined_actions.csv"
)
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, f"macd_checkpoint_pop{POP_SIZE}_iters{MAX_ITERATIONS}_{SEARCH_MODE}.pkl")


def pool_initializer(ohlcv_np):
    ohlcv_initializer(ohlcv_np)


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
            func = signal_func_mapping[best_params["signal_type"]]
            signals = np.array(func(macd, macd_signal))
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
            RESULTS_DIR, f"macd_pop{POP_SIZE}_iters{MAX_ITERATIONS}_mode{SEARCH_MODE}.csv"
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
