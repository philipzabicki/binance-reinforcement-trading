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

from common import save_checkpoint, load_checkpoint
from definitions import REPORT_DIR
from genetic_classes.feature_action_fitter import StochasticOscillatorFitting
from utils.feature_generation import ohlcv_initializer, custom_StochasticOscillator

# from utils.miscellaneous import convert_variables
from utils.ta_tools import extract_segments_indices
from utils.ta_tools import (
    k_int_cross,
    k_ext_cross,
    d_int_cross,
    d_ext_cross,
    k_cross_int_os_ext_ob,
    k_cross_ext_os_int_ob,
    d_cross_int_os_ext_ob,
    d_cross_ext_os_int_ob,
    kd_cross,
    kd_cross_inside,
    kd_cross_outside,
)

CPU_CORES_COUNT = 20
print(f"CPUs used: {CPU_CORES_COUNT}")

TERMINATION = DefaultMultiObjectiveTermination(
    ftol=0.001, period=5, n_max_gen=100, n_max_evals=1_000_000
)

PROBLEM = StochasticOscillatorFitting
ALGORITHM = MixedVariableGA
POP_SIZE = 8192
MAX_ITERATIONS = 35

RESULTS_DIR = os.path.join(REPORT_DIR, "feature_fits")
ACTIONS_FULLPATH = os.path.join(
    REPORT_DIR, "optimal_actions", "final_combined_actions.csv"
)
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "StochOsc_checkpoint.pkl")


def pool_initializer(np_df):
    ohlcv_initializer(np_df)


def main(max_iterations=10):
    df = pd.read_csv(ACTIONS_FULLPATH)
    np_df = df.to_numpy()[:, 1:6].astype(float)
    target_actions = df["Action"].values

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
        unmatched_segments = extract_segments_indices(target_actions)
        all_results = []
        print(f"Initial unmatched segments: {len(unmatched_segments)}")

    with Pool(CPU_CORES_COUNT, initializer=pool_initializer, initargs=(np_df,)) as pool:
        runner = StarmapParallelization(pool.starmap)
        while unmatched_segments.shape[0]:
            print(
                f"Iteration {iteration}, unmatched segments: {len(unmatched_segments)}"
            )

            problem = PROBLEM(
                target_segments=unmatched_segments, elementwise_runner=runner
            )

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

            slowK, slowD = custom_StochasticOscillator(
                np_df,
                fastK_period=best_params["fastK_period"],
                slowK_period=best_params["slowK_period"],
                slowD_period=best_params["slowD_period"],
                slowK_ma_type=best_params["slowK_ma_type"],
                slowD_ma_type=best_params["slowD_ma_type"],
            )

            signal_func_mapping = {
                "k_int_cross": k_int_cross,
                "k_ext_cross": k_ext_cross,
                "d_int_cross": d_int_cross,
                "d_ext_cross": d_ext_cross,
                "k_cross_int_os_ext_ob": k_cross_int_os_ext_ob,
                "k_cross_ext_os_int_ob": k_cross_ext_os_int_ob,
                "d_cross_int_os_ext_ob": d_cross_int_os_ext_ob,
                "d_cross_ext_os_int_ob": d_cross_ext_os_int_ob,
                "kd_cross": kd_cross,
                "kd_cross_inside": kd_cross_inside,
                "kd_cross_outside": kd_cross_outside,
            }

            func = signal_func_mapping[best_params["signal_type"]]
            signals = np.array(
                func(
                    k_line=slowK,
                    d_line=slowD,
                    oversold_threshold=best_params["oversold_threshold"],
                    overbought_threshold=best_params["overbought_threshold"],
                )
            )
            gen_segments = extract_segments_indices(signals)

            gen_segments_set = {tuple(seg) for seg in gen_segments}
            # matched = [seg for seg in unmatched_segments if tuple(seg) in gen_segments_set]
            matched = sum(
                1 for seg in unmatched_segments if tuple(seg) in gen_segments_set
            )
            if matched:
                print(f"Matched segments in iteration {iteration}: {matched}")
                unmatched_segments = np.array(
                    [
                        seg
                        for seg in unmatched_segments
                        if tuple(seg) not in gen_segments_set
                    ]
                )
                all_results.append(
                    {"iteration": iteration, "params": best_params, "matched": matched}
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
            RESULTS_DIR, f"stoch_osc_pop{POP_SIZE}_iters{MAX_ITERATIONS}.csv"
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
