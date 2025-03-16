import os

# import numpy as np
from multiprocessing import Pool, cpu_count

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

from definitions import REPORT_DIR
from genetic_classes.feature_action_fitter import StochasticOscillatorFitting
from utils.miscellaneous import convert_variables

CPU_CORES_COUNT = cpu_count()
print(f"CPUs used: {CPU_CORES_COUNT}")

PROBLEM = StochasticOscillatorFitting
# SIGNALS_SOURCE = 'cross'
# SIGNALS_SOURCE = 'mid-cross'
SIGNALS_SOURCE = "threshold"

ALGORITHM = MixedVariableGA
POP_SIZE = 2048
TERMINATION = ("n_gen", 150)

# RESULTS_FILENAME = 'stoch_osc_cross.csv'
# RESULTS_FILENAME = 'stoch_osc_mid_cross.csv'
RESULTS_FILENAME = "stoch_osc_threshold.csv"
RESULTS_DIR = os.path.join(REPORT_DIR, "feature_fits")
ACTIONS_FULLPATH = os.path.join(
    REPORT_DIR, "optimal_actions", "final_combined_actions.csv"
)


def main():
    df = pd.read_csv(ACTIONS_FULLPATH)
    print(f"df used:  {df}")
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

    results = []  # Lista do przechowywania wyników dla danego ma_type
    for lower, upper in zip(quantiles[:-1], quantiles[1:]):
        print(f"Optimize run for Stochastic Oscillator, range: ({lower}, {upper})")
        problem = PROBLEM(
            df, lower, upper, signals_source=SIGNALS_SOURCE, elementwise_runner=runner
        )

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
            best_gene = {"lower": lower, "upper": upper, "reward": -res.F[0]}
            best_gene.update(variables_dict)
            print(f"Best gene: {best_gene}")
            results.append(best_gene)
        else:
            for front, var in zip(res.F, res.X):
                variables_dict = convert_variables(var)
                pareto_result = {"lower": lower, "upper": upper, "reward": -front}
                pareto_result.update(variables_dict)
                print(f"Pareto front: {pareto_result}")
                results.append(pareto_result)
        output_file = os.path.join(RESULTS_DIR, RESULTS_FILENAME)
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
