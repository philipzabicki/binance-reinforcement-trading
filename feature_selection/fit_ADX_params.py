import os
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

from common import save_checkpoint, load_checkpoint, build_bin_df, _equal_freq_assignments
from definitions import REPORT_DIR
from genetic_classes.feature_action_fitter import ADXFitting
from utils.feature_generation import (
    custom_ADX,
    adx_initializer,
)
from utils.ta_tools import (extract_segments_indices,
                            ADX_trend_signal,
                            ADX_DIs_cross_above_threshold,
                            ADX_DIs_approaching_cross_above_threshold)

PROBLEM = ADXFitting
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
    mutation={Real: PM(eta=7),
              Integer: PM(eta=7, vtype=float, repair=RoundingRepair())
              },
    crossover={Real: SBX(eta=3),
               Integer: SBX(eta=3, vtype=float, repair=RoundingRepair())
               },
    eliminate_duplicates=MixedVariableDuplicateElimination())

CPU_CORES_COUNT = 16
print(f"CPUs used: {CPU_CORES_COUNT}")
POP_SIZE = 8192
MAX_ITERATIONS = 10
N_SPLITS = 3
SEARCH_MODE = 'mix'

RESULTS_DIR = os.path.join(REPORT_DIR, "feature_fits_quick")
ACTIONS_FULLPATH = os.path.join(
    REPORT_DIR, "optimal_actions", "final_combined_actions.csv"
)
CHECKPOINT_FILE = os.path.join(RESULTS_DIR,
                               f"ADX_checkpoint_pop{POP_SIZE}_iters{MAX_ITERATIONS}_{SEARCH_MODE}.pkl")


def pool_initializer(ohlcv_np):
    adx_initializer(ohlcv_np)


def run_part(df_h,
             half_idx,
             ohlcv_np,
             max_iterations):
    checkpoint_file = CHECKPOINT_FILE.replace(".pkl", f"_h{half_idx}.pkl")
    output_file = os.path.join(
        RESULTS_DIR,
        f"adx_pop{POP_SIZE}_iters{max_iterations}_mode{SEARCH_MODE}_h{half_idx}.csv"
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
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
            print(f'Best parameters found: {best_params}')

            adx, plus_DI, minus_DI = custom_ADX(
                ohlcv_np,
                atr_period=best_params["atr_period"],
                posDM_period=best_params["posDM_period"],
                negDM_period=best_params["negDM_period"],
                adx_period=best_params["adx_period"],
                ma_type_atr=best_params["ma_type_atr"],
                ma_type_posDM=best_params["ma_type_posDM"],
                ma_type_negDM=best_params["ma_type_negDM"],
                ma_type_adx=best_params["ma_type_adx"],
            )

            signal_func_mapping = {
                "ADX_trend_signal": ADX_trend_signal,
                "ADX_DIs_cross_above_threshold": ADX_DIs_cross_above_threshold,
                "ADX_DIs_approaching_cross_above_threshold": ADX_DIs_approaching_cross_above_threshold,
            }
            signals = np.array(
                signal_func_mapping[best_params["signal_type"]](adx, plus_DI, minus_DI, best_params["adx_threshold"]))
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


def main(n_splits=2, max_iterations=MAX_ITERATIONS):
    df = pd.read_csv(ACTIONS_FULLPATH)
    ohlcv_np = df.to_numpy()[:, 1:6].astype(float)
    print(f'All segments {int((df["Action"] != 0).sum()) // 2}')

    if n_splits == 0:
        print("n_splits=0 -> using the full dataframe (no zeroing).")
        run_part(df.copy(), half_idx=0, ohlcv_np=ohlcv_np, max_iterations=max_iterations)
        return

    assignments = _equal_freq_assignments(df["Weight"], n_splits)

    sizes = [int((assignments == i).sum()) for i in range(n_splits)]
    n_inactive = int((assignments == -1).sum())
    print(f"Created {n_splits} bins. Sizes (active only): {sizes} | inactive (Weight==0): {n_inactive}")

    for b_idx in range(n_splits):
        df_b = build_bin_df(df, b_idx, assignments)
        # df_b.to_csv(f'{b_idx}.csv', index=False)
        buys = int((df_b["Action"] == 1).sum())
        sells = int((df_b["Action"] == -1).sum())
        print(f"\n=== Bin {b_idx} – buys: {buys} sells: {sells} ===")
        w = df_b.loc[df_b["Weight"] != 1.0, "Weight"]
        print(f"=== Weight mean: {w.mean()} max: {w.max()} min: {w.min()} ===")
        run_part(df_b, half_idx=b_idx, ohlcv_np=ohlcv_np, max_iterations=max_iterations)


if __name__ == "__main__":
    start_time = time.time()  # start timing
    main(n_splits=N_SPLITS, max_iterations=MAX_ITERATIONS)
    end_time = time.time()  # end timing
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    # profile_main()
