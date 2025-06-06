import os
from csv import writer
from os import makedirs, path, mkdir

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from datetime import datetime as dt
from numpy import (
    median,
    sum,
    percentile,
    inf,
    array,
    min,
    mean,
    max,
    ndarray,
    nan_to_num,
    float32,
)
from scipy.stats import hmean, gmean
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from pymoo.core.callback import Callback
from pymoo.core.variable import Integer, Real, Choice
from pymoo.visualization.pcp import PCP

from definitions import REPORT_DIR


def save_results_action(filename, result):
    filename = REPORT_DIR + filename + ".csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as file:
        # _header = ['reward'] + [*result.pop.get("X")[0].keys()]
        _header = ["reward", "action_sequence"]  # for action optimizer
        writer(file).writerow(_header)
        for f, x in zip(result.pop.get("F"), result.pop.get("X")):
            _row = [-1 * f[0], *x]
            writer(file).writerow(_row)
            # print(f'writing row {_row}')


def save_results(filename, result):
    filename = REPORT_DIR + filename + ".csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as file:
        # _header = ['reward'] + [*result.pop.get("X")[0].keys()]
        _header = ["reward", "action_sequence"]  # for action optimizer
        writer(file).writerow(_header)
        for f, x in zip(result.pop.get("F"), result.pop.get("X")):
            _row = [-1 * f[0], *x]
            writer(file).writerow(_row)
            # print(f'writing row {_row}')


def get_callback_plot(callback, fname, dpi=75):
    plt.figure(figsize=(16, 9))
    plt.title("Convergence")
    # plt.ylabel('Reward (min/avg/max)')
    plt.ylabel("Reward metric (min/avg/max)")
    plt.xlabel("Population No.")
    plt.plot(-1 * array(callback.opt), "--")
    plt.savefig(REPORT_DIR + "Convergence_" + fname + ".png", dpi=dpi)
    return plt


def get_variables_plot(x_variables, problem, fname, save=True, dpi=75):
    sample_gen = x_variables[0]
    if not isinstance(sample_gen, dict):
        raise NotImplementedError(
            "Currently supports only list of dict type populations"
        )
    # print(f'x_variables {x_variables}')
    X_array = array([list(map(float, entry.values())) for entry in x_variables])
    # print(f'X_array {X_array}')
    pop_size = len(X_array)
    # Handling chart displayed boundaries for different variable types
    labels = [*sample_gen.keys()]
    bounds = []
    for name in labels:
        if isinstance(problem.vars[name], (Integer, Real)):
            bounds.append(problem.vars[name].bounds)
        elif isinstance(problem.vars[name], Choice):
            bounds.append(
                (min(problem.vars[name].options), max(problem.vars[name].options))
            )
            # print(f'choices.options {problem.vars[name].options}')
    bounds = array(bounds).T
    # print(bounds)
    # bounds = array([problem.vars[name].bounds for name in labels]).T
    # bounds = array([problem.vars[name].bounds for name in labels if hasattr(problem.vars[name], 'bounds')]).T
    plot = PCP(figsize=(21, 9), labels=labels, bounds=bounds, tight_layout=True)
    plot.set_axis_style(color="grey", alpha=1)
    plot.add(X_array, color="grey", alpha=0.3)
    plot.add(X_array[int(pop_size * 0.9) + 1:], linewidth=1.9, color="#a4f0ff")
    plot.add(
        X_array[int(pop_size * 0.8) + 1: int(pop_size * 0.9)],
        linewidth=1.8,
        color="#88e7fa",
    )
    plot.add(
        X_array[int(pop_size * 0.7) + 1: int(pop_size * 0.8)],
        linewidth=1.7,
        color="#60d8f3",
    )
    plot.add(
        X_array[int(pop_size * 0.6) + 1: int(pop_size * 0.7)],
        linewidth=1.6,
        color="#33c5e8",
    )
    plot.add(
        X_array[int(pop_size * 0.5) + 1: int(pop_size * 0.6)],
        linewidth=1.5,
        color="#12b0da",
    )
    plot.add(
        X_array[int(pop_size * 0.4) + 1: int(pop_size * 0.5)],
        linewidth=1.4,
        color="#019cc8",
    )
    plot.add(
        X_array[int(pop_size * 0.3) + 1: int(pop_size * 0.4)],
        linewidth=1.3,
        color="#0086b4",
    )
    plot.add(
        X_array[int(pop_size * 0.2) + 1: int(pop_size * 0.3)],
        linewidth=1.2,
        color="#00719f",
    )
    plot.add(
        X_array[int(pop_size * 0.1) + 1: int(pop_size * 0.2)],
        linewidth=1.1,
        color="#005d89",
    )
    plot.add(X_array[: int(pop_size * 0.1)], linewidth=1.0, color="#004a73")
    plot.add(X_array[0], linewidth=1.5, color="red")
    if save:
        plot.save(REPORT_DIR + fname + ".png", dpi=dpi)
    return plot


class VariablesPlotCallback(Callback):
    def __init__(self, problem) -> None:
        super().__init__()
        self.problem = problem
        self.opt = []
        self.gen_n = 0
        _date = str(dt.today()).replace(":", "-")[:-7]
        self.file_location = f"frames/{self.problem.env.__class__.__name__}_{_date}/"
        dir = REPORT_DIR + self.file_location
        makedirs(path.dirname(dir), exist_ok=True)

    def notify(self, algorithm):
        filename = f"{self.file_location}/{self.gen_n}"
        get_variables_plot(
            algorithm.pop.get("X"), self.problem, filename, save=True, dpi=25
        )
        self.gen_n += 1
        avg_rew = mean(algorithm.pop.get("F"))
        if avg_rew < 0:
            min_rew = min(algorithm.pop.get("F"))
            max_rew = max(algorithm.pop.get("F"))
            self.opt.append([min_rew, avg_rew, max_rew])
        else:
            self.opt.append([0.0, 0.0, 0.0])


class GenerationSavingCallback(Callback):
    def __init__(self, problem, dir_name, verbose=False) -> None:
        super().__init__()
        self.verbose = verbose
        self.problem = problem
        self.opt = []
        self.gen_n = 0
        full_path = REPORT_DIR + dir_name
        mkdir(full_path)
        self.dir_name = dir_name

    def notify(self, algorithm):
        filepath = f"{self.dir_name}/{self.gen_n}"
        get_variables_plot(
            algorithm.pop.get("X"), self.problem, filepath, save=True, dpi=50
        )
        save_results(filepath, algorithm)
        if self.verbose:
            print(
                f'Best gen: reward={-algorithm.pop.get("F")[0]} vars={algorithm.pop.get("X")[0]}'
            )
        # print(f'algorithm.pop.get("F") {algorithm.pop.get("F")}')
        # print(f'algorithm.pop.get("X") {algorithm.pop.get("X")}')
        avg_rew = mean(algorithm.pop.get("F"))
        if avg_rew < 0:
            min_rew = min(algorithm.pop.get("F"))
            max_rew = max(algorithm.pop.get("F"))
            self.opt.append([min_rew, avg_rew, max_rew])
        else:
            self.opt.append([0.0, 0.0, 0.0])
        self.gen_n += 1


class MinAvgMaxNonzeroSingleObjCallback(Callback):
    def __init__(self, problem, verbose=False) -> None:
        super().__init__()
        self.opt = []
        self.problem = problem
        self.verbose = verbose

    def notify(self, algorithm):
        if self.verbose:
            print(
                f'Best gen: reward={-algorithm.pop.get("F")[0]} vars={algorithm.pop.get("X")[0]}'
            )
        avg_rew = mean(algorithm.pop.get("F"))
        if avg_rew < 0:
            min_rew = min(algorithm.pop.get("F"))
            max_rew = max(algorithm.pop.get("F"))
            self.opt.append([min_rew, avg_rew, max_rew])
        else:
            self.opt.append([0.0, 0.0, 0.0])


class AverageNonzeroSingleObjCallback(Callback):
    def __init__(self, problem, verbose=False) -> None:
        super().__init__()
        self.opt = []
        self.problem = problem
        self.verbose = verbose

    def notify(self, algorithm):
        if self.verbose:
            print(
                f'Best gen: reward={-algorithm.pop.get("F")[0]} vars={algorithm.pop.get("X")[0]}'
            )
        avg_rew = mean(algorithm.pop.get("F"))
        if avg_rew < 0:
            self.opt.append(avg_rew)
        else:
            self.opt.append(0.0)


def reward_from_metric(rewards: ndarray, n_evals: int, metric: str) -> float:
    if metric == "median":
        return median(rewards)
    elif metric == "first_quartile":
        # print(f'rewards {rewards} 1Q {percentile(rewards, 25)}')
        return percentile(nan_to_num(rewards.astype(float32)), 75)
    elif metric == "mean":
        return mean(rewards)
    elif metric == "max":
        return max(rewards)
    elif metric == "min":
        return min(rewards)
    elif metric == "robust_mean":
        if inf in rewards:
            return inf
        else:
            scaled_rews = RobustScaler().fit_transform(rewards.reshape(n_evals, -1))
            return mean(scaled_rews)
    elif metric == "robust_sum":
        if inf in rewards:
            return inf
        else:
            scaled_rews = RobustScaler().fit_transform(rewards.reshape(n_evals, -1))
            return sum(scaled_rews)
    elif metric == "minmax_mean":
        if inf in rewards:
            return inf
        else:
            scaled_rews = MinMaxScaler(feature_range=(-1, 1)).fit_transform(
                rewards.reshape(n_evals, -1)
            )
            return mean(scaled_rews)
    elif metric == "minmax_sum":
        if inf in rewards:
            return inf
        else:
            scaled_rews = MinMaxScaler(feature_range=(-1, 1)).fit_transform(
                rewards.reshape(n_evals, -1)
            )
            return sum(scaled_rews)
    # Supports only positive values
    elif metric == "hmean":
        if inf in rewards:
            return inf
        else:
            scaled_rews = MinMaxScaler(feature_range=(1, 100)).fit_transform(
                rewards.reshape(n_evals, -1)
            )
            return hmean(scaled_rews)[0]
    # Supports only positive values
    elif metric == "gmean":
        if inf in rewards:
            return inf
        else:
            scaled_rews = MinMaxScaler(feature_range=(1, 100)).fit_transform(
                rewards.reshape(n_evals, -1)
            )
            return gmean(scaled_rews)[0]
    elif metric == "median_mean_mix":
        _median = median(rewards)
        _mean = mean(rewards)
        if (_median < 0) and (_mean < 0):
            return (_median + _mean) / 2
        return _median
    elif metric == "median_min_mix":
        _median = median(rewards)
        min_rew = min(rewards)
        if (_median < 0) and (min_rew < 0):
            return (_median + min_rew) / 2
        return _median
    elif metric == "quartile_min_mix":
        third_quartile = percentile(nan_to_num(rewards.astype(float32)), 75)
        min_rew = min(rewards)
        if (third_quartile < 0) and (min_rew < 0):
            return (third_quartile + min_rew) / 2
        return third_quartile
    elif metric == "sign_minmax_sum":
        positive_rews = rewards[(rewards > 0) & (rewards != inf)]
        positive_rews_len = len(positive_rews)
        if positive_rews_len > 1:
            positive_rews = MinMaxScaler(feature_range=(0, 1)).fit_transform(
                positive_rews.reshape(positive_rews_len, -1)
            )
        elif positive_rews_len == 1:
            positive_rews = array([1])
        else:
            positive_rews = array([0])
        negative_rews = rewards[(rewards < 0) & (rewards != -inf)]
        negative_rews_len = len(negative_rews)
        if negative_rews_len > 1:
            negative_rews = MinMaxScaler(feature_range=(-1, 0)).fit_transform(
                negative_rews.reshape(negative_rews_len, -1)
            )
        elif negative_rews_len == 1:
            negative_rews = array([-1])
        else:
            negative_rews = array([0])
        return sum(positive_rews) + sum(negative_rews)
