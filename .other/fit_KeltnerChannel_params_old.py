import os
from multiprocessing import Pool, cpu_count

import pandas as pd
# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.mixed import MixedVariableGA
# import matplotlib.pyplot as plt
from pymoo.core.mixed import (
    MixedVariableMating,
    MixedVariableSampling,
    MixedVariableDuplicateElimination,
)
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize

from definitions import REPORT_DIR
from genetic_classes.feature_action_fitter import (
    KeltnerChannelFitting,
)
from utils.miscellaneous import convert_variables

CPU_CORES_COUNT = cpu_count()
print(f"CPUs used: {CPU_CORES_COUNT}")

PROBLEM = KeltnerChannelFitting
ALGORITHM = MixedVariableGA
POP_SIZE = 1024
TERMINATION = ("n_gen", 75)

RESULTS_DIR = os.path.join(REPORT_DIR, "feature_fits", "ma_band_action_fits")
ACTIONS_FULLPATH = os.path.join(
    REPORT_DIR, "optimal_actions", "final_combined_actions.csv"
)


def main():
    df = pd.read_csv(ACTIONS_FULLPATH)
    # vals = df.loc[df['Weight'] > 1.0]
    # print(f'vals {vals}')
    quantiles = list(
        pd.qcut(df.loc[df["Weight"] > 1.0]["Weight"], q=10, labels=False, retbins=True)[
            1
        ]
    )
    print("Granice binów:")
    print(quantiles)

    # bins = pd.cut(df.loc[df['Weight'] > 1.0]['Weight'], q=10, include_lowest=True, right=True)
    # bin_counts = bins.value_counts().sort_index()
    # print(f'bin_counts {bin_counts}')

    pool = Pool(CPU_CORES_COUNT)
    runner = StarmapParallelization(pool.starmap)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for ma_type in range(15, 32):
        results = []  # Lista do przechowywania wyników dla danego ma_type
        for lower, upper in zip(quantiles[:-1], quantiles[1:]):
            print(f"Optimize run for ma_type: {ma_type}, range: ({lower}, {upper})")
            problem = PROBLEM(df, lower, upper, ma_type, elementwise_runner=runner)

            algorithm = ALGORITHM(
                n_jobs=-1,
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
                # callback=GenerationSavingCallback(problem, dir_name, verbose=True),
                # callback=MinAvgMaxNonzeroSingleObjCallback(problem, verbose=True),
                termination=TERMINATION,
                verbose=True,
            )

            if len(res.F) == 1:
                variables_dict = convert_variables(res.X)
                best_gene = {
                    "ma_type": ma_type,
                    "lower": lower,
                    "upper": upper,
                    "reward": -res.F[0],
                }
                best_gene.update(variables_dict)
                print(f"Best gene: {best_gene}")
                results.append(best_gene)
            else:
                for front, var in zip(res.F, res.X):
                    variables_dict = convert_variables(var)
                    pareto_result = {
                        "ma_type": ma_type,
                        "lower": lower,
                        "upper": upper,
                        "reward": -front,
                    }
                    pareto_result.update(variables_dict)
                    print(f"Pareto front: {pareto_result}")
                    results.append(pareto_result)
        if results:
            df_results = pd.DataFrame(results)
            output_file = os.path.join(RESULTS_DIR, f"results_ma_type_{ma_type}.csv")
            df_results.to_csv(output_file, index=False)
            print(f"Results for ma_type {ma_type} saved to {output_file}")
        else:
            print(f"No results to save for ma_type {ma_type}")
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
