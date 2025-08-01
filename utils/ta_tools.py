# from typing import Any
# from matplotlib import pyplot as plt
from math import copysign
from statistics import mean, stdev
from time import time

# import Levenshtein
import numpy as np
import pandas as pd
import pandas_ta as p_ta
import ta.momentum
import ta.trend
import talib
from finta import TA as finTA
from numba import jit
from tindicators import ti

from utils import get_data
from .custom_ma import (
    HullMA,
    VWMA,
    RMA,
    ALMA,
    HammingMA,
    LWMA,
    MGD,
    GMA,
    FBA,
    NadarayWatsonMA,
)

np.seterr(over="raise")


# np.seterr(divide='ignore', invalid='ignore')
# from scipy import stats
# config.THREADING_LAYER = 'safe'
# import MAs
# import datetime as dt
# from numba import njit, jit
# from scipy.ndimage import convolve1d as conv
# from sklearn import preprocessing


def feature_timeit(feature_func: callable) -> callable:
    """
    Measure the execution time of a TA feature function.

    Args:
        feature_func (Callable): The feature function to be timed.
    Returns:
        Callable: A wrapped version of the feature function that measures and
        prints the execution time.
    """

    def wrapper(*args, **kwargs):
        start_t = time()
        print(f"\r adding {feature_func.__name__} feature...", end="")
        ret = feature_func(*args, **kwargs)
        print(f" ({(time() - start_t) * 1_000:.2f}ms)")
        return ret

    return wrapper


@feature_timeit
def scale_columns(df: pd.DataFrame, scaler: callable) -> pd.DataFrame:
    """
    Scales TA indicators inside provided dataframe.

    Args:
        df (pd.DataFrame): dataframe of TA indicators values to scale
        scaler (callable): sklearn preprocessing scaler to be used ex. MinMaxScaler()
    Returns:
        pd.DataFrame: scaled dataframe

    It skips OHLC and date/time columns.
    """
    for col in df.columns:
        if col not in [
            "Open time",
            "Opened",
            "Close time",
            "Open",
            "High",
            "Low",
            "Close",
        ]:
            # print(col)
            # scaler.fit(df[[col]])
            df[col] = scaler.fit_transform(df[[col]])
    return df


@feature_timeit
@jit(nopython=True, nogil=True, cache=True)
def zscore_standardize(values: list | np.ndarray) -> np.ndarray:
    """
    (...)
    """
    return (values - np.mean(values)) / np.std(values)


# # @jit(nopython=True, nogil=True, cache=True)
# def extract_segments_indices(signal):
#     segments = []
#     n = len(signal)
#     i = 0
#     while i < n:
#         if signal[i] == 1:
#             start = i
#             i += 1
#             while i < n and signal[i] != -1:
#                 i += 1
#             if i < n:  # Found -1
#                 segments.append((start, i))
#         else:
#             i += 1
#     return segments
@jit(nopython=True, nogil=True, cache=True)
def extract_segments_indices(signal):
    n = signal.shape[0]
    segments_out = np.empty((n // 2, 2), dtype=np.int64)
    seg_count = 0
    i = 0
    while i < n:
        if signal[i] == 1:
            start = i
            i += 1
            while i < n and signal[i] != -1:
                i += 1
            if i < n:  # Found -1
                segments_out[seg_count, 0] = start
                segments_out[seg_count, 1] = i
                seg_count += 1
        else:
            i += 1
    return segments_out[:seg_count]


# def profit_weighted_sequence_score_lib_old(optimal_signals, generated_signals, weights, buy=1, sell=-1, inactivity=0):
#     total_weight = 0.0
#     sequence_score_sum = 0.0
#     n = len(optimal_signals)
#     i = 0
#     while i < n:
#         if optimal_signals[i] == buy:
#             seq_start = i
#             seq_end = None
#             for j in range(i + 1, n):
#                 if optimal_signals[j] == sell:
#                     seq_end = j
#                     break
#             if seq_end is None:
#                 seq_end = n - 1
#             optimal_seq = ''.join(str(x) for x in optimal_signals[seq_start:seq_end + 1])
#             generated_seq = ''.join(str(x) for x in generated_signals[seq_start:seq_end + 1])
#
#             ed = Levenshtein.distance(optimal_seq, generated_seq)
#             norm_ed = ed / (seq_end - seq_start + 1)
#             seq_weight = np.sum(weights[seq_start:seq_end + 1])
#             sequence_score_sum += seq_weight * (1.0 - norm_ed)
#             total_weight += seq_weight
#             i = seq_end + 1
#         else:
#             if optimal_signals[i] == inactivity:
#                 seq_weight = weights[i]
#                 score = 1.0 if generated_signals[i] == inactivity else 0.0
#                 sequence_score_sum += seq_weight * score
#                 total_weight += seq_weight
#             i += 1
#     return sequence_score_sum / total_weight if total_weight > 0 else 0.0


def weighted_levenshtein_match_combo_score(optimal_signals, generated_signals, optimal_signals_segments,
                                           generated_signals_segments, weights, buy=1, sell=-1, inactivity=0):
    total_weight = 0.0
    sequence_score_sum = 0.0
    perfect_matches = 0
    sequences_count = 0
    weights = np.array(weights)

    for seg in optimal_signals_segments:
        seq_start, seq_end = seg[0], seg[1]
        ed = np.count_nonzero(optimal_signals[seq_start:seq_end + 1] != generated_signals[seq_start:seq_end + 1])
        norm_ed = ed / (seq_end - seq_start + 1)
        seq_weight = np.sum(weights[seq_start:seq_end + 1])
        print(f'optimal_seq {optimal_signals[seq_start:seq_end + 1]}')
        print(f'generated_seq {generated_signals[seq_start:seq_end + 1]}')
        print(f'ed {ed} norm_ed {norm_ed} seq_weight {seq_weight}')

        sequence_score_sum += seq_weight * (1.0 - norm_ed)
        total_weight += seq_weight

        sequences_count += 1
        if ed == 0:
            perfect_matches += 1

    used = np.zeros(optimal_signals.shape[0], dtype=bool)
    for seg in optimal_signals_segments:
        used[seg[0]:seg[1] + 1] = True

    for i in range(optimal_signals.shape[0]):
        if not used[i] and optimal_signals[i] == inactivity:
            seq_weight = weights[i]
            score = 1.0 if generated_signals[i] == inactivity else 0.0
            sequence_score_sum += seq_weight * score
            total_weight += seq_weight

    base_score = sequence_score_sum / total_weight if total_weight > 0 else 0.0
    print(f'base_score {base_score}')
    if sequences_count > 0:
        perfect_match_ratio = perfect_matches / sequences_count
        adjusted_score = base_score * perfect_match_ratio
    else:
        adjusted_score = base_score

    return adjusted_score


# Calculates and returns linear regression slope but predictor variable(X) are natural numbers from 1 to len of dependent variable(Y)
# Y are supposed to be balance divided by initial balance ratios per every env step
@jit(nopython=True, nogil=True, cache=True)
def linear_reg_slope(Y):
    Y = np.array(Y)
    n = len(Y)
    X = np.arange(1, n + 1)
    # print(f'X: {X}')
    x_mean = np.mean(X)
    Sxy = np.sum(X * Y) - n * x_mean * np.mean(Y)
    Sxx = np.sum(X * X) - n * x_mean ** 2
    return Sxy / Sxx


@jit(nopython=True, nogil=True, cache=True)
def linear_slope_indicator(values: list | np.ndarray) -> float:
    """
    Computes a linear slope indicator based on a subset of values.

    Args:
        values (list | np.ndarray): Input values for computing the slope indicator.
        This can be either a NumPy array or a list of numerical values.

    Returns:
        float: The computed linear slope indicator.

    The function computes the linear slope indicator using a subset of the input values.
    It calculates the slope/s using a specified portions of the input values and then
    applies a some transformation to the result before returning it.

    Example:
        >>> some_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> indicator = linear_slope_indicator(some_values)
        >>> print(indicator)
        1.0
    """
    _5 = len(values) // 20
    percentile25 = linear_reg_slope(values[-_5 * 5:])
    # percentile50 = linear_reg_slope(values[-_5*10:])
    # percentile75 = linear_reg_slope(values[-_5*15:])
    # percentile95 = linear_reg_slope(values[-_5*19:])
    slope_avg = percentile25
    # return slope_avg
    return copysign(abs(slope_avg) ** (1 / 4), slope_avg)


######################################################################################
#                                SIGNAL GENERATORS                                   #
######################################################################################
# @feature_timeit
@jit(nopython=True, nogil=True, cache=True)
def RSI_like_signal(
        rsi_like_indicator: list[float] | np.ndarray,
        timeperiod: int,
        top_bound: float = 80.0,
        bottom_bound: float = 20.0,
) -> list[float]:
    """
    Calculate RSI like (Ultimate oscillator, MFI) signals for a given column of values.

    Args:
        top_bound:
        bottom_bound:
        rsi_like_indicator (np.ndarray or List[float]): An array or list of RSI like values [0,100].
        timeperiod (int): The time period for calculating the macd_env.py.

    Returns:
        List[float]: A list of signals corresponding to the input values [-1,1].

    Input values interpretation:
      >=90, singal: -0.75, extremely overbought
      >=75, macd_env.py: -0.5, overbought
      >=65, macd_env.py: -0.25, somewhat overbought
      <=10, macd_env.py: 0.75, extremely oversold
      <=25, macd_env.py: 0.5, oversold
      <=35, macd_env.py: 0.25, somewhat oversold
      For 80/20 line crosses:
      80.0 line cross from above, macd_env.py: -1, bearish cross
      20.0 line cross from below, macd_env.py: 1, bullish cross

    Example:
        >>> rsi_values = [60.0, 55.0, 45.0, 35.0, 19.0, 40.0, 75.0, 85.0]
        >>> RSI_like_signal(rsi_values, 3)
        [None, None, None, 0.25, 0.5, 1, -0.25, -0.5]
    """
    return [0.0] * timeperiod + [
        (
            -1
            if (prev > top_bound) and (cur <= top_bound)
            else (
                1
                if (prev < bottom_bound) and (cur >= bottom_bound)
                else (
                    -0.75
                    if (cur > 90.0)
                    else (
                        0.75
                        if (cur < 10.0)
                        else (
                            -0.5
                            if (cur > 75.0)
                            else (
                                0.5
                                if (cur < 25.0)
                                else (
                                    -0.25
                                    if (cur > 65.0)
                                    else 0.25 if (cur < 35.0) else 0
                                )
                            )
                        )
                    )
                )
            )
        )
        for cur, prev in zip(
            rsi_like_indicator[timeperiod:], rsi_like_indicator[timeperiod - 1: -1]
        )
    ]


@jit(nopython=True, nogil=True, cache=True)
def RSI_oversold_signal(
        rsi_like_indicator: list[float] | np.ndarray,
        timeperiod: int,
        oversold_threshold: float = 20.0,
) -> list[float]:
    return [0.0] * timeperiod + [
        1 if (prev > oversold_threshold) and (cur <= oversold_threshold) else 0
        for cur, prev in zip(
            rsi_like_indicator[timeperiod:], rsi_like_indicator[timeperiod - 1: -1]
        )
    ]


@jit(nopython=True, nogil=True, cache=True)
def StochasticOscillator_threshold_cross_signal(
        k_line: list[float] | np.ndarray,
        d_line: list[float] | np.ndarray,
        oversold_threshold: float = 20.0,
        overbought_threshold: float = 80.0,
) -> list[float]:
    """
    Zwraca listę sygnałów:
      *  +1  gdy K i D < oversold_threshold i wystąpiło przecięcie z dołu do góry (K > D przy jednoczesnym prev_K < prev_D)
      *  -1  gdy K i D > overbought_threshold i wystąpiło przecięcie z góry na dół (K < D przy jednoczesnym prev_K > prev_D)
      *   0  w przeciwnym wypadku
    """
    return [0.0] + [
        (
            1.0
            if (
                    (cur_k < oversold_threshold)
                    and (cur_d < oversold_threshold)
                    and (cur_k > cur_d)
                    and (prev_k < prev_d)
            )
            else (
                -1.0
                if (
                        (cur_k > overbought_threshold)
                        and (cur_d > overbought_threshold)
                        and (cur_k < cur_d)
                        and (prev_k > prev_d)
                )
                else 0.0
            )
        )
        for cur_k, cur_d, prev_k, prev_d in zip(
            k_line[1:], d_line[1:], k_line[:-1], d_line[:-1]
        )
    ]


@jit(nopython=True, nogil=True, cache=True)
def StochasticOscillator_mid_cross_signal(
        k_line: list[float] | np.ndarray,
        d_line: list[float] | np.ndarray,
        mid_level: float = 50.0,
) -> list[float]:
    """
    Zwraca listę sygnałów:
      *  +0.75 gdy K i D < mid_level i nastąpiło przecięcie z dołu do góry
      *  -0.75 gdy K i D > mid_level i nastąpiło przecięcie z góry na dół
      *   0    w przeciwnym wypadku
    """
    return [0.0] + [
        (
            1.0
            if (
                    (cur_k < mid_level)
                    and (cur_d < mid_level)
                    and (cur_k > cur_d)
                    and (prev_k < prev_d)
            )
            else (
                -1.0
                if (
                        (cur_k > mid_level)
                        and (cur_d > mid_level)
                        and (cur_k < cur_d)
                        and (prev_k > prev_d)
                )
                else 0.0
            )
        )
        for cur_k, cur_d, prev_k, prev_d in zip(
            k_line[1:], d_line[1:], k_line[:-1], d_line[:-1]
        )
    ]


@jit(nopython=True, nogil=True, cache=True)
def StochasticOscillator_threshold_signal(
        k_line: list[float] | np.ndarray,
        d_line: list[float] | np.ndarray,
        oversold_threshold: float = 20.0,
        overbought_threshold: float = 80.0,
) -> list[float]:
    """
    Zwraca listę sygnałów w zależności od tego, czy aktualne wartości K i D
    znajdują się powyżej/poniżej ustalonych progów:
      * +0.5  gdy K i D < oversold_threshold
      * -0.5  gdy K i D > overbought_threshold
      * +0.25 gdy K lub D < oversold_threshold (jeśli warunek wyżej nie jest spełniony)
      * -0.25 gdy K lub D > overbought_threshold (jeśli warunek wyżej nie jest spełniony)
      *  0     w przeciwnym wypadku
    """
    return [0.0] + [
        (
            1.0
            if (cur_k < oversold_threshold) and (cur_d < oversold_threshold)
            else (
                -1.0
                if (cur_k > overbought_threshold) and (cur_d > overbought_threshold)
                else (
                    0.5
                    if (cur_k < oversold_threshold) or (cur_d < oversold_threshold)
                    else (
                        -0.5
                        if (cur_k > overbought_threshold)
                           or (cur_d > overbought_threshold)
                        else 0.0
                    )
                )
            )
        )
        for cur_k, cur_d in zip(k_line[1:], d_line[1:])
    ]


########################################################################################################################
# New ones for Stoch Osc
# 1. k_int_cross: k-line internal threshold cross
@jit(nopython=True, nogil=True, cache=True)
def k_int_cross(
        k_line: list[float] | np.ndarray,
        d_line: list[float] | np.ndarray = None,
        oversold_threshold: float = 20.0,
        overbought_threshold: float = 80.0,
) -> list[float]:
    return [0.0] + [
        (
            1.0
            if (prev > oversold_threshold > curr)
            else -1.0 if (prev < overbought_threshold < curr) else 0.0
        )
        for curr, prev in zip(k_line[1:], k_line[:-1])
    ]


# 2. k_ext_cross: k-line external threshold cross
@jit(nopython=True, nogil=True, cache=True)
def k_ext_cross(
        k_line: list[float] | np.ndarray,
        d_line: list[float] | np.ndarray = None,
        oversold_threshold: float = 20.0,
        overbought_threshold: float = 80.0,
) -> list[float]:
    return [0.0] + [
        (
            1.0
            if (prev < oversold_threshold < curr)
            else -1.0 if (prev > overbought_threshold > curr) else 0.0
        )
        for curr, prev in zip(k_line[1:], k_line[:-1])
    ]


# 3. d_int_cross: d-line internal threshold cross
@jit(nopython=True, nogil=True, cache=True)
def d_int_cross(
        k_line: list[float] | np.ndarray,
        d_line: list[float] | np.ndarray = None,
        oversold_threshold: float = 20.0,
        overbought_threshold: float = 80.0,
) -> list[float]:
    return [0.0] + [
        (
            1.0
            if (prev > oversold_threshold > curr)
            else -1.0 if (prev < overbought_threshold < curr) else 0.0
        )
        for curr, prev in zip(d_line[1:], d_line[:-1])
    ]


# 4. d_ext_cross: d-line external threshold cross
@jit(nopython=True, nogil=True, cache=True)
def d_ext_cross(
        k_line: list[float] | np.ndarray,
        d_line: list[float] | np.ndarray = None,
        oversold_threshold: float = 20.0,
        overbought_threshold: float = 80.0,
) -> list[float]:
    return [0.0] + [
        (
            1.0
            if (prev < oversold_threshold < curr)
            else -1.0 if (prev > overbought_threshold > curr) else 0.0
        )
        for curr, prev in zip(d_line[1:], d_line[:-1])
    ]


# 5. k_cross_int_os_ext_ob:
#    Buy signal when k-line crosses oversold internally; sell signal when k-line crosses overbought externally.
@jit(nopython=True, nogil=True, cache=True)
def k_cross_int_os_ext_ob(
        k_line: list[float] | np.ndarray,
        d_line: list[float] | np.ndarray = None,
        oversold_threshold: float = 20.0,
        overbought_threshold: float = 80.0,
) -> list[float]:
    return [0.0] + [
        (
            1.0
            if (prev > oversold_threshold > curr)
            else -1.0 if (prev > overbought_threshold > curr) else 0.0
        )
        for curr, prev in zip(k_line[1:], k_line[:-1])
    ]


# 6. k_cross_ext_os_int_ob:
#    Buy signal when k-line crosses oversold externally; sell signal when k-line crosses overbought internally.
@jit(nopython=True, nogil=True, cache=True)
def k_cross_ext_os_int_ob(
        k_line: list[float] | np.ndarray,
        d_line: list[float] | np.ndarray = None,
        oversold_threshold: float = 20.0,
        overbought_threshold: float = 80.0,
) -> list[float]:
    return [0.0] + [
        (
            1.0
            if (prev < oversold_threshold < curr)
            else -1.0 if (prev < overbought_threshold < curr) else 0.0
        )
        for curr, prev in zip(k_line[1:], k_line[:-1])
    ]


# 7. d_cross_int_os_ext_ob:
#    Buy signal when d-line crosses oversold internally; sell signal when d-line crosses overbought externally.
@jit(nopython=True, nogil=True, cache=True)
def d_cross_int_os_ext_ob(
        k_line: list[float] | np.ndarray,
        d_line: list[float] | np.ndarray = None,
        oversold_threshold: float = 20.0,
        overbought_threshold: float = 80.0,
) -> list[float]:
    return [0.0] + [
        (
            1.0
            if (prev > oversold_threshold > curr)
            else -1.0 if (prev > overbought_threshold > curr) else 0.0
        )
        for curr, prev in zip(d_line[1:], d_line[:-1])
    ]


# 8. d_cross_ext_os_int_ob:
#    Buy signal when d-line crosses oversold externally; sell signal when d-line crosses overbought internally.
@jit(nopython=True, nogil=True, cache=True)
def d_cross_ext_os_int_ob(
        k_line: list[float] | np.ndarray,
        d_line: list[float] | np.ndarray = None,
        oversold_threshold: float = 20.0,
        overbought_threshold: float = 80.0,
) -> list[float]:
    return [0.0] + [
        (
            1.0
            if (prev < oversold_threshold < curr)
            else -1.0 if (prev < overbought_threshold < curr) else 0.0
        )
        for curr, prev in zip(d_line[1:], d_line[:-1])
    ]


# 9. kd_cross: Generic crossing between k and d lines (no threshold limits)
@jit(nopython=True, nogil=True, cache=True)
def kd_cross(
        k_line: list[float] | np.ndarray,
        d_line: list[float] | np.ndarray = None,
        oversold_threshold: float = 20.0,
        overbought_threshold: float = 80.0,
) -> list[float]:
    return [0.0] + [
        (
            1.0
            if (prev_k < prev_d and curr_k > curr_d)
            else -1.0 if (prev_k > prev_d and curr_k < curr_d) else 0.0
        )
        for curr_k, curr_d, prev_k, prev_d in zip(
            k_line[1:], d_line[1:], k_line[:-1], d_line[:-1]
        )
    ]


# 10. kd_cross_inside: k and d cross when both are within the threshold range (between oversold and overbought)
@jit(nopython=True, nogil=True, cache=True)
def kd_cross_inside(
        k_line: list[float] | np.ndarray,
        d_line: list[float] | np.ndarray = None,
        oversold_threshold: float = 20.0,
        overbought_threshold: float = 80.0,
) -> list[float]:
    return [0.0] + [
        (
            1.0
            if (
                    prev_k < prev_d
                    and curr_k > curr_d
                    and (oversold_threshold < curr_k < overbought_threshold)
                    and (oversold_threshold < curr_d < overbought_threshold)
            )
            else (
                -1.0
                if (
                        prev_k > prev_d
                        and curr_k < curr_d
                        and (oversold_threshold < curr_k < overbought_threshold)
                        and (oversold_threshold < curr_d < overbought_threshold)
                )
                else 0.0
            )
        )
        for curr_k, curr_d, prev_k, prev_d in zip(
            k_line[1:], d_line[1:], k_line[:-1], d_line[:-1]
        )
    ]


# 11. kd_cross_outside: k and d cross when both are outside the threshold range on the same side
@jit(nopython=True, nogil=True, cache=True)
def kd_cross_outside(
        k_line: list[float] | np.ndarray,
        d_line: list[float] | np.ndarray = None,
        oversold_threshold: float = 20.0,
        overbought_threshold: float = 80.0,
) -> list[float]:
    return [0.0] + [
        (
            1.0
            if (
                    prev_k < prev_d
                    and curr_k > curr_d
                    and (curr_k < oversold_threshold and curr_d < oversold_threshold)
            )
            else (
                -1.0
                if (
                        prev_k > prev_d
                        and curr_k < curr_d
                        and (
                                curr_k > overbought_threshold and curr_d > overbought_threshold
                        )
                )
                else 0.0
            )
        )
        for curr_k, curr_d, prev_k, prev_d in zip(
            k_line[1:], d_line[1:], k_line[:-1], d_line[:-1]
        )
    ]


# TODO: Split/add threshold for buy/sell signal
# @feature_timeit
@jit(nopython=True, nogil=True, cache=True)
def ADX_trend_signal(
        adx_col, minus_di, plus_di, adx_threshold
):
    return [1 if (adx >= adx_threshold) and (pDI > mDI)
            else
            -1 if (adx >= adx_threshold) and (pDI < mDI)
            else 0
            for adx, mDI, pDI in zip(adx_col, minus_di, plus_di)]


# TODO: Split/add threshold for buy/sell signal
# @feature_timeit
@jit(nopython=True, nogil=True, cache=True)
def ADX_DIs_cross_above_threshold(
        adx_col, minus_di, plus_di, adx_threshold
):
    return [0.0] + [
        0 if adx < adx_threshold else
        1 if (cur_pDI > cur_mDI) and (prev_pDI < prev_mDI) else
        -1 if (cur_pDI < cur_mDI) and (prev_pDI > prev_mDI) else
        0
        for cur_pDI, cur_mDI, adx, prev_pDI, prev_mDI in zip(
            plus_di[1:], minus_di[1:], adx_col[1:], plus_di[:-1], minus_di[:-1]
        )
    ]


# TODO: Split/add threshold for buy/sell signal
# @feature_timeit
@jit(nopython=True, nogil=True, cache=True)
def ADX_DIs_approaching_cross_above_threshold(
        adx_col, minus_di, plus_di, adx_threshold
):
    return [0.0] + [
        0 if adx < adx_threshold else
        1 if (cur_pDI > prev_pDI) and (cur_mDI < prev_mDI) and (cur_pDI < cur_mDI) else
        -1 if (cur_pDI < prev_pDI) and (cur_mDI > prev_mDI) and (cur_pDI > cur_mDI) else
        0
        for cur_pDI, cur_mDI, adx, prev_pDI, prev_mDI in zip(
            plus_di[1:], minus_di[1:], adx_col[1:], plus_di[:-1], minus_di[:-1]
        )
    ]


# @feature_timeit
@jit(nopython=True, nogil=True, cache=True)
def ADX_signal_old(
        adx_col: list | np.ndarray, minus_di: list | np.ndarray, plus_di: list | np.ndarray
) -> list[float]:
    """
    Calculate ADX (Average Directional Index) signals based on the given parameters.

    Args:
        adx_col (np.ndarray or List[float]): An array of ADX values.
        minus_di (np.ndarray or List[float]): An array of minus DI values.
        plus_di (np.ndarray or List[float]): An array of plus DI values.

    Returns:
        List[float]: A list of ADX signals calculated according to the specified conditions:

        0.0 when adx is below 25.0
        1.0 buliish DI cross and adx above 35.0
        0.75 same but adx only above 25.0
        -1.0 bearish DI cross and adx above 25.0
        -0.75 same but adx only above 25.0

        0.5 DIs are getting closer to bullish cross
        -0.5 DIs are getting closer to bearish cross
        0.25 getting closer to bullish cross but with adx only above 25.0:
        -0.25 getting closer to bearish cross but with adx only above 25.0:
        0.0 for other cases

    Note:
        The input arrays must have the same length.
        First value in returned list is always 0.0

    Example:
        >>> adx_values = np.array([30.0, 28.0, 26.0, 25.0])
        >>> minusDI_values = np.array([10.0, 12.0, 14.0, 16.0])
        >>> plusDI_values = np.array([20.0, 18.0, 16.0, 14.0])
        >>> ADX_signal(adx_values, minusDI_values, plusDI_values)
        [0.0, -0.5, -0.5, -1.0]
    """
    return [0.0] + [
        (
            0.0
            if adx < 25.0
            else (
                1
                if (cur_pDI > cur_mDI) and (prev_pDI < prev_mDI) and (adx >= 35.0)
                else (
                    0.75
                    if (cur_pDI > cur_mDI) and (prev_pDI < prev_mDI) and (adx >= 25.0)
                    else (
                        -1
                        if (cur_pDI < cur_mDI)
                           and (prev_pDI > prev_mDI)
                           and (adx >= 35.0)
                        else (
                            -0.75
                            if (cur_pDI < cur_mDI)
                               and (prev_pDI > prev_mDI)
                               and (adx >= 25.0)
                            else (
                                0.5
                                if (cur_pDI > prev_pDI)
                                   and (cur_mDI < prev_mDI)
                                   and (cur_pDI < cur_mDI)
                                   and (adx >= 35.0)
                                else (
                                    -0.5
                                    if (cur_pDI < prev_pDI)
                                       and (cur_mDI > prev_mDI)
                                       and (cur_pDI > cur_mDI)
                                       and (adx >= 35.0)
                                    else (
                                        0.25
                                        if (cur_pDI > prev_pDI)
                                           and (cur_mDI < prev_mDI)
                                           and (cur_pDI < cur_mDI)
                                           and (adx >= 25.0)
                                        else (
                                            -0.25
                                            if (cur_pDI < prev_pDI)
                                               and (cur_mDI > prev_mDI)
                                               and (cur_pDI > cur_mDI)
                                               and (adx >= 25.0)
                                            else 0.0
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        for cur_pDI, cur_mDI, adx, prev_pDI, prev_mDI in zip(
            plus_di[1:], minus_di[1:], adx_col[1:], plus_di[:-1], minus_di[:-1]
        )
    ]


# @feature_timeit
@jit(nopython=True, nogil=True, cache=True)
def ADX_trend_signal_old(
        adx_col, minus_di, plus_di, adx_threshold
):
    """
    Returns:
        List[float]: A list of trend signals. Possible values:
            1: Strong uptrend (ADX >= 75.0 and plus DI > minus DI)
            -1: Strong downtrend (ADX >= 75.0 and plus DI < minus DI)
            0.75: Moderate uptrend (ADX >= 50.0 and plus DI > minus DI)
            -0.75: Moderate downtrend (ADX >= 50.0 and plus DI < minus DI)
            0.5: Weak uptrend (ADX >= 25.0 and plus DI > minus DI)
            -0.5: Weak downtrend (ADX >= 25.0 and plus DI < minus DI)
            0.25: Very weak uptrend (ADX < 25.0 and plus DI > minus DI)
            -0.25: Very weak downtrend (ADX < 25.0 and plus DI < minus DI)
            0: No clear trend.
    Example:
        >>> ADX = [70.0, 80.0, 60.0]
        >>> minusDI = [20.0, 30.0, 40.0]
        >>> plusDI = [40.0, 50.0, 30.0]
        >>> ADX_trend_signal(ADX, minusDI, plusDI)
        [0.75, 1.0, -0.75]
    """
    return [
        (
            1
            if (adx >= 75.0) and (pDI > mDI)
            else (
                -1
                if (adx >= 75.0) and (pDI < mDI)
                else (
                    0.75
                    if (adx >= 50.0) and (pDI > mDI)
                    else (
                        -0.75
                        if (adx >= 50.0) and (pDI < mDI)
                        else (
                            0.5
                            if (adx >= 25.0) and (pDI > mDI)
                            else (
                                -0.5
                                if (adx >= 25.0) and (pDI < mDI)
                                else (
                                    0.25
                                    if (adx < 25.0) and (pDI > mDI)
                                    else -0.25 if (adx < 25.0) and (pDI < mDI) else 0
                                )
                            )
                        )
                    )
                )
            )
        )
        for adx, mDI, pDI in zip(adx_col, minus_di, plus_di)
    ]


@jit(nopython=True, nogil=True, cache=True)
def MACD_lines_cross_with_zero(
        macd_col: list | np.ndarray, signal_col: list | np.ndarray
) -> list[float | int]:
    return [0.0] + [
        (
            1
            if (cur_sig < 0)
               and (cur_macd < 0)
               and (cur_macd > cur_sig)
               and (prev_macd < prev_sig)
            else (
                -1
                if (cur_sig > 0)
                   and (cur_macd > 0)
                   and (cur_macd < cur_sig)
                   and (prev_macd > prev_sig)
                else 0
            )
        )
        for cur_sig, cur_macd, prev_sig, prev_macd in zip(
            signal_col[1:], macd_col[1:], signal_col[:-1], macd_col[:-1]
        )
    ]


@jit(nopython=True, nogil=True, cache=True)
def MACD_lines_cross(
        macd_col: list | np.ndarray, signal_col: list | np.ndarray
) -> list[float | int]:
    return [0.0] + [
        (
            1
            if (cur_macd > cur_sig) and (prev_macd < prev_sig)
            else -1 if (cur_macd < cur_sig) and (prev_macd > prev_sig) else 0
        )
        for cur_sig, cur_macd, prev_sig, prev_macd in zip(
            signal_col[1:], macd_col[1:], signal_col[:-1], macd_col[:-1]
        )
    ]


@jit(nopython=True, nogil=True, cache=True)
def MACD_lines_approaching_cross_with_zero(
        macd_col: list | np.ndarray, signal_col: list | np.ndarray
) -> list[float | int]:
    return [0.0] + [
        (
            1
            if (cur_sig < 0)
               and (cur_macd < 0)
               and (cur_macd > prev_macd)
               and (cur_sig < prev_sig)
            else (
                -1
                if (cur_sig > 0)
                   and (cur_macd > 0)
                   and (cur_macd < prev_macd)
                   and (cur_sig > prev_sig)
                else 0
            )
        )
        for cur_sig, cur_macd, prev_sig, prev_macd in zip(
            signal_col[1:], macd_col[1:], signal_col[:-1], macd_col[:-1]
        )
    ]


@jit(nopython=True, nogil=True, cache=True)
def MACD_lines_approaching_cross(
        macd_col: list | np.ndarray, signal_col: list | np.ndarray
) -> list[float | int]:
    return [0.0] + [
        (
            1
            if (cur_macd > prev_macd) and (cur_sig < prev_sig)
            else -1 if (cur_macd < prev_macd) and (cur_sig > prev_sig) else 0
        )
        for cur_sig, cur_macd, prev_sig, prev_macd in zip(
            signal_col[1:], macd_col[1:], signal_col[:-1], macd_col[:-1]
        )
    ]


@jit(nopython=True, nogil=True, cache=True)
def MACD_signal_line_zero_cross(_, signal_col: list | np.ndarray) -> list[float | int]:
    return [0.0] + [
        1 if cur_sig > 0 > prev_sig else -1 if cur_sig < 0 < prev_sig else 0
        for cur_sig, prev_sig in zip(signal_col[1:], signal_col[:-1])
    ]


@jit(nopython=True, nogil=True, cache=True)
def MACD_line_zero_cross(macd_col: list | np.ndarray, _) -> list[float | int]:
    return [0.0] + [
        1 if cur_macd > 0 > prev_macd else -1 if cur_macd < 0 < prev_macd else 0
        for cur_macd, prev_macd in zip(macd_col[1:], macd_col[:-1])
    ]


@jit(nopython=True, nogil=True, cache=True)
def MACD_histogram_reversal(
        macd_col: list | np.ndarray, signal_col: list | np.ndarray
) -> list[float | int]:
    macdhist_col = macd_col - signal_col
    return [0.0] * 3 + [
        (
            1.0
            if cur_macd > prev_macd and prev_macd < preprev_macd < prepreprev_macd
            else (
                -1.0
                if cur_macd < prev_macd and prev_macd > preprev_macd > prepreprev_macd
                else 0
            )
        )
        for cur_macd, prev_macd, preprev_macd, prepreprev_macd in zip(
            macdhist_col[3:], macdhist_col[2:-1], macdhist_col[1:-2], macdhist_col[:-3]
        )
    ]


# @feature_timeit
@jit(nopython=True, nogil=True, cache=True)
def MACD_histogram_reversal_signal(
        macdhist_col: list | np.ndarray,
) -> list[float | int]:
    """
    Calculate MACD histogram reversal signals.

    Args:
        macdhist_col (np.ndarray or List[float]): An array or list of MACD histogram values.

    Returns:
        List[float]: A list of reversal signals, where 1 represents a bullish reversal,
        -1 represents a bearish reversal, and 0 represents no reversal.

    The function calculates MACD histogram reversal signals based on the MACD histogram values provided in the input.
    It compares the current MACD histogram value with the previous three values to determine whether a bullish
    or bearish reversal has occurred.

    Note:
    - If cur_macd > prev_macd and prev_macd < preprev_macd < prepreprev_macd, it's a bullish reversal(returns 1).
    - If cur_macd < prev_macd and prev_macd > preprev_macd > prepreprev_macd, it's a bearish reversal(returns -1).
    - Otherwise, no reversal is detected (returns 0).

    Example:
    >>> macdhist_values = [0.5, 0.6, 0.4, 0.3, 0.8, 0.9, 1.0, 0.7]
    >>> signals = MACDhist_reversal_signal(macdhist_values)
    >>> print(signals)
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
    """
    return [0.0] * 3 + [
        (
            1.0
            if cur_macd > prev_macd and prev_macd < preprev_macd < prepreprev_macd
            else (
                -1.0
                if cur_macd < prev_macd and prev_macd > preprev_macd > prepreprev_macd
                else 0
            )
        )
        for cur_macd, prev_macd, preprev_macd, prepreprev_macd in zip(
            macdhist_col[3:], macdhist_col[2:-1], macdhist_col[1:-2], macdhist_col[:-3]
        )
    ]


# @feature_timeit
@jit(nopython=True, nogil=True, cache=True)
def MACD_cross_signal(
        macd_col: list | np.ndarray, signal_col: list | np.ndarray
) -> list[float | int]:
    """
    Calculate MACD (Moving Average Convergence Divergence) crossover signals.

    Args:
        macd_col (np.ndarray or List[float]): An array or list containing MACD values.
        signal_col (np.ndarray or List[float]): An array or list containing macd_env.py line values.

    Returns:
        List[float]: A list of crossover signals with values 1, -1, 0.5, or -0.5.

    The function calculates crossover signals based on the provided MACD and macd_env.py data. It compares
    the current MACD and macd_env.py values with the previous values and assigns a macd_env.py.

    Note:
    The input arrays or lists should have the same length.

    Example:
    >>> macd_values = [12.5, 14.2, 11.8, 15.6, 9.8]
    >>> signal_values = [10.3, 12.6, 11.2, 13.9, 10.3]
    >>> signals = MACD_cross_signal(macd_values, signal_values)
    >>> print(signals)
    [0, 0, 0, 0, -1]
    """
    return [0.0] + [
        (
            1
            if (cur_sig < 0)
               and (cur_macd < 0)
               and (cur_macd > cur_sig)
               and (prev_macd < prev_sig)
            else (
                0.75
                if (cur_macd > cur_sig) and (prev_macd < prev_sig)
                else (
                    -1
                    if (cur_sig > 0)
                       and (cur_macd > 0)
                       and cur_macd < cur_sig
                       and prev_macd > prev_sig
                    else (
                        -0.75
                        if cur_macd < cur_sig and prev_macd > prev_sig
                        else (
                            0.5
                            if (cur_sig < 0)
                               and (cur_macd < 0)
                               and cur_macd > prev_macd
                               and cur_sig < prev_sig
                            else (
                                -0.5
                                if (cur_sig > 0)
                                   and (cur_macd > 0)
                                   and cur_macd < prev_macd
                                   and cur_sig > prev_sig
                                else (
                                    0.25
                                    if cur_macd > prev_macd and cur_sig < prev_sig
                                    else (
                                        -0.25
                                        if cur_macd < prev_macd and cur_sig > prev_sig
                                        else 0
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        for cur_sig, cur_macd, prev_sig, prev_macd in zip(
            signal_col[1:], macd_col[1:], signal_col[:-1], macd_col[:-1]
        )
    ]


# @feature_timeit
@jit(nopython=True, nogil=True, cache=True)
def MACD_zero_cross_signal(
        macd_col: list | np.ndarray, signal_col: list | np.ndarray
) -> list[float | int]:
    """
    Calculate the zero crossing values for MACD (Moving Average Convergence Divergence).

    This function takes two lists or arrays, `macd_col` and `signal_col`, and computes zero crossing values for MACD.

    Zero crossing is determined based on the relationship between current and previous values in both columns.

    Args:
        macd_col (np.ndarray or List[float]): An array or list containing MACD values.
        signal_col (np.ndarray or List[float]): An array or list containing macd_env.py line values.

    Returns:
        List[Union[float, int]]: A list of zero crossing singals [-1,1]

    Signals are generated on the following criteria:
      - 0.5 for MACD going from negative to positive
      - 1 for Signal line going from negative to positive
      - 0.5 for MACD going from positive to negative
      - -1 for Signal line going from positive to negative
      - 0 for no zero crossing

    Note:
        - The length of `macd_col` and `signal_col` should be the same.
        - The function assumes that both lists have at least 2 elements.

    Examples:
        >>> macd = [1.0, 0.5, -0.2, -0.7, 0.9]
        >>> macd_env.py = [0.5, 0.1, -0.3, -0.6, 0.8]
        >>> MACD_zerocross_signal(macd, macd_env.py)
        [0, 0.5, -0.5, 0, 1]

    """
    return [0.0] + [
        (
            1
            if cur_sig > 0 > prev_sig
            else (
                -1
                if cur_sig < 0 < prev_sig
                else (
                    0.5
                    if cur_macd > 0 > prev_macd
                    else -0.5 if cur_macd < 0 < prev_macd else 0
                )
            )
        )
        for cur_macd, cur_sig, prev_macd, prev_sig in zip(
            macd_col[1:], signal_col[1:], macd_col[:-1], signal_col[:-1]
        )
    ]


# @feature_timeit
def ChaikinOscillator_signal(
        chaikin_oscillator: list | np.ndarray,
) -> list[float | int]:
    return [0.0] + [
        1 if cur_chosc > 0 > prev_chosc else -1 if cur_chosc < 0 < prev_chosc else 0
        for cur_chosc, prev_chosc in zip(
            chaikin_oscillator[1:], chaikin_oscillator[:-1]
        )
    ]


@feature_timeit
def BB_signal(
        close: list | np.ndarray,
        upper_band: list | np.ndarray,
        mid: list | np.ndarray,
        lower_band: list | np.ndarray,
) -> list[float]:
    """
    Calculate Bollinger Bands macd_env.py based on the provided data.

    Args:
        close (np.ndarray or List[float]): A one-dimensional array of closing prices.
        upper_band (np.ndarray or List[float]): A one-dimensional array of upper Bollinger Bands values.
        mid (np.ndarray or List[float]): A one-dimensional array of middle Bollinger Bands values.
        lower_band (np.ndarray or List[float]): A one-dimensional array of lower Bollinger Bands values.

    Returns:
        List[int]: A list of signals ranging from strong buy (1) to strong sell (-1).

    Note:
        - Buy (1) when the current closing price is below the lower Bollinger Band.
        - Weak buy (0.5) when the current closing price crossed mid from below.
        - Sell (-1) when the current closing price is above the upper Bollinger Band.
        - Weak sell (-0.5) when the current closing price crossed mid from above.
        - Hold (0) in other cases.

    The function calculates Bollinger Bands signals for each pair of consecutive data points and
    returns a list of signals.

    Example:
        close_prices = [50.0, 52.0, 48.0, 56.5, 57.0]
        upper_bands = [52.0, 54.0, 50.0, 57.0, 55.0]
        mid_bands = [51.0, 53.0, 49.0, 56.0, 54.0]
        lower_bands = [50.0, 52.0, 48.0, 55.0, 53.0]

        signals = BB_signal(close_prices, upper_bands, mid_bands, lower_bands)
        # Output: [1, 1, 1, 0.5, -1]

    """
    return [0.0] + [
        (
            -1
            if currC >= Ub
            else (
                1
                if currC <= Lb
                else (
                    -0.5
                    if (currC < Mid) and (preC > Mid)
                    else 0.5 if (currC > Mid) and (preC < Mid) else 0
                )
            )
        )
        for currC, preC, Lb, Mid, Ub in zip(
            close[1:], close[:-1], lower_band[1:], mid[1:], upper_band[1:]
        )
    ]


@feature_timeit
def price_levels(
        open: list | np.ndarray,
        close: list | np.ndarray,
        decimals: int = 0,
        sort: bool = False,
) -> list[int]:
    """
    Calculate price levels based on Open and Close prices.

    Args:
        open (list | np.ndarray[float]): An array or list of opening prices.
        close (list | np.ndarray[float]): An array or list of closing prices.
        decimals (int, optional): Number of decimal places to round price-level to. Defaults to 0.
        sort (bool, optional): Whether to sort the price-levels by frequency. Defaults to False.

    Returns:
        List[int]: A list of integer values representing the frequency of price levels.

    The function calculates price levels based on the Open and Close prices. Price levels are determined
    by checking if the price moved up or down between two consecutive periods and rounding the Close
    price to the specified number of decimals. The resulting price-levels are then counted, and the
    function returns a list of integers representing the frequency of each price level.

    If `sort` is set to True, the price levels are sorted in descending order of frequency.
    """

    def get_levels(open, close, decimals=0, sort=False):
        price_levels = {}
        for open, close, close_next in zip(open[:-1], close[:-1], close[1:]):
            if (close > open > close_next) or (close < open < close_next):
                lvl = round(close, decimals)
                if lvl in price_levels:
                    price_levels[lvl] += 1
                else:
                    price_levels[lvl] = 1
        if sort:
            price_levels = {
                k: v
                for k, v in sorted(
                    price_levels.items(), key=lambda item: item[1], reverse=True
                )
            }
        return price_levels

    levels = get_levels(open, close, decimals=decimals, sort=sort)
    return [
        levels[round(c, decimals)] if round(c, decimals) in levels.keys() else 0
        for c in close
    ]


@feature_timeit
def move_probability(open: list | np.ndarray, close: list | np.ndarray) -> list[float]:
    """
    Calculate the probability of a price move at given moment based on distribution (by standardizing)
    of historical open and close data.

    Args:
        open (np.ndarray or list[float]): An array or list of opening prices.
        close (np.ndarray or list[float]): An array or list of closing prices.

    Returns:
        List[float]: A list of move probabilities for each data point.

    The function calculates the probability of a move by comparing the Open and Close prices.
    It uses the following formula:

    probability = ((price_change - avg_gain) / gain_stddev) if close > open,
                    else ((price_change - avg_loss) / loss_stddev)

    where price_change is the percentage change in price, avg_gain is the average gain,
    gain_stddev is the standard deviation of gains, avg_loss is the average loss, and
    loss_stddev is the standard deviation of losses.

    Note: This function assumes that Open and Close arrays/lists have the same length.
    """

    def get_avg_changes(open, close):
        gain = [(c / o - 1) * 100 for o, c in zip(open, close) if c > o]
        loss = [(o / c - 1) * 100 for o, c in zip(open, close) if c < o]
        return mean(gain), stdev(gain), mean(loss), stdev(loss)

    avg_gain, gain_stddev, avg_loss, loss_stddev = get_avg_changes(open, close)
    return [
        (
            (((c / o - 1) * 100) - avg_gain) / gain_stddev
            if c > o
            else (((o / c - 1) * 100) - avg_loss) / loss_stddev
        )
        for o, c in zip(open, close)
    ]


@feature_timeit
def hourly_seasonality_by_ticker(
        ticker: str, opened_as_dt_hour: pd.Series, type: str = "um", data: str = "klines"
) -> list[float]:
    """
    Calculate hourly seasonality for a given ticker.

    Args:
        ticker (str): The ticker symbol. (e.g., 'BTCUSDT')
        opened_as_dt_hour (pd.Series): A pandas Series containing the datetime hour for each data point. [0,24)
        type (str, optional): The type of data 'um' (USDs-M) or 'cm' (COIN-M) for futures, 'spot' for spot.
                              Defaults to 'um'.
        data (str, optional): The data source: {  aggTrades, bookDepth, bookTicker, indexPriceKlines, klines,
                                                  liquidationSnapshot, markPriceKlines, metrics, premiumIndexKlines,
                                                  trades  }.
                              Defaults to 'klines'.

    Returns:
        List[float]: A list of hourly seasonality values for the given ticker.

    This function calculates the hourly seasonality for a given ticker based on the provided hour series.
    It first retrieves data using the specified parameters and then computes the average percentage change for each hour of the day (0-23).
    Finally, it returns a list of hourly seasonality values corresponding to the provided opening hours.

    Note:
    - The function internally uses the `get_data.by_BinanceVision` function to retrieve data for the ticker.

    """

    def get_hour_changes(ticker="BTCUSDT", market_type="um", data_type="klines"):
        df = get_data.by_BinanceVision(ticker, "1h", market_type=type, data_type=data)
        hours = df[f"Opened"].dt.hour.to_numpy()
        co_change = ((df["Close"].to_numpy() / df["Open"].to_numpy()) - 1) * 100
        return {
            i: np.mean([chng for chng, h in zip(co_change, hours) if h == i])
            for i in range(0, 24)
        }

    h_dict = get_hour_changes(ticker, type, data)
    return [h_dict[h] for h in opened_as_dt_hour]


@feature_timeit
def hourly_seasonality(df: pd.DataFrame) -> list[float]:
    """
    Calculates the hourly seasonality of stock price changes.

    Args:
        df (pd.DataFrame): A DataFrame containing price data.
            It should have a 'Opened' column with datetime values and 'Close' and 'Open' column
            with corresponding prices.

    Returns:
        List[float]: A list of hourly seasonality values, representing the average
            percentage change in stock price for each hour of the day (0-23).
    """
    hours = df["Opened"].dt.hour
    co_change = ((df["Close"] / df["Open"]) - 1) * 100
    # get mean change for each hour(i) in day
    h_dict = {
        i: np.mean([chng for chng, h in zip(co_change, hours) if h == i])
        for i in range(0, 24)
    }
    return [h_dict[h] for h in hours]


@feature_timeit
def daily_seasonality_by_ticker(
        ticker: str, opened_as_dt_weekday: pd.Series, type: str = "um", data: str = "klines"
) -> list[float]:
    """
    Compute the daily seasonality for a given ticker.

    Args:
        ticker (str): The ticker symbol, e.g., 'BTCUSDT'.
        opened_as_dt_weekday (pd.Series): A Pandas Series containing the weekdays for each data point. [0,7)
        type (str, optional): The type of data 'um' (USDs-M) or 'cm' (COIN-M) for futures, 'spot' for spot.
                              Defaults to 'um'.
        data (str, optional): The data source: {  aggTrades, bookDepth, bookTicker, indexPriceKlines, klines,
                                                  liquidationSnapshot, markPriceKlines, metrics, premiumIndexKlines,
                                                  trades  }.
                              Defaults to 'klines'.

    Returns:
        List[float]: A list of daily seasonality values corresponding to each weekday.

    Example:
    >>> ticker = 'BTCUSDT'
    >>> weekdays = pd.Series([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])  # Example weekday data
    >>> seasonality = daily_seasonality_by_ticker(ticker, weekdays)

    Note:
    - This function calculates the daily seasonality as the mean change (in percentage) from Open to Close
      for a given ticker, grouped by each day of the week (0 to 6, where 0 is Monday and 6 is Sunday).
    - The function internally uses the `get_data.by_BinanceVision` function to retrieve data for the ticker.
    """

    def get_weekday_changes(ticker="BTCUSDT", market_type="um", data_type="klines"):
        df = get_data.by_BinanceVision(ticker, "1d", market_type=type, data_type=data)
        weekdays = df["Opened"].dt.dayofweek.to_numpy()
        co_change = ((df["Close"].to_numpy() / df["Open"].to_numpy()) - 1) * 100
        return {
            i: np.mean([chng for chng, day in zip(co_change, weekdays) if day == i])
            for i in range(0, 7)
        }

    wd_dict = get_weekday_changes(ticker, type, data)
    return [wd_dict[w] for w in opened_as_dt_weekday]


@feature_timeit
def daily_seasonality(df: pd.DataFrame) -> list[float]:
    weekdays = df["Opened"].dt.dayofweek
    co_change = ((df["Close"] / df["Open"]) - 1) * 100
    wd_dict = {
        i: np.mean([chng for chng, day in zip(co_change, weekdays) if day == i])
        for i in range(0, 7)
    }
    return [wd_dict[w] for w in weekdays]


######################################################################################
######################################################################################
######################################################################################

######################################################################################
#                             Non typical MA functions                               #
######################################################################################

# @feature_timeit


######################################################################################
######################################################################################
######################################################################################


# @feature_timeit
def get_MA(np_df: np.ndarray, ma_type: int, ma_period: int) -> np.ndarray:
    """
    Calculate Moving Average (MA) based on the specified MA type and period.

    Parameters:
    - np_df (np.ndarray): Numpy array containing OHLCV (Open, High, Low, Close, Volume) data.
    - ma_type (int): Type of Moving Average to calculate. Choose from the following options:
        0: Simple Moving Average (SMA)
        1: Exponential Moving Average (EMA)
        2: Weighted Moving Average (WMA)
        3: Kaufman's Adaptive Moving Average (KAMA)
        4: Triangular Moving Average (TRIMA)
        5: Double Exponential Moving Average (DEMA)
        6: Triple Exponential Moving Average (TEMA)
        7: Triple Exponential Moving Average (T3)
        8: MESA Adaptive Moving Average (MAMA)
        9: Linear Regression Moving Average (LINEARREG)
        10: Simple Moving Median (SMM)
        11: Smoothed Simple Moving Average (SSMA)
        12: Volume Adjusted Moving Average (VAMA)
        13: Zero Lag Exponential Moving Average (ZLEMA)
        14: Exponential Volume Weighted Moving Average (EVWMA)
        15: Smoothed Moving Average (SMMA)
        16: Volume Weighted Moving Average (VWMA)
        17: Symmetrically Weighted Moving Average (SWMA) - Ascending
        18: Symmetrically Weighted Moving Average (SWMA) - Descending
        19: Exponential Hull Moving Average (EHMA)
        20: Leo Moving Average (LMA)
        21: Sharp Modified Moving Average (SHMMA)
        22: Ahrens Moving Average (AHMA)
        23: Hull Moving Average (HullMA)
        24: Volume Weighted Moving Average (VWMA)
        25: Relative Moving Average (RMA)
        26: Arnaud Legoux Moving Average (ALMA)
        27: Hamming Moving Average (HammingMA)
        28: Linear Weighted Moving Average (LWMA)
        29: McGinley Dynamic (MGD)
        30: Geometric Moving Average (GMA)
        31: Fibonacci Based Average (FBA)
        32: Nadaray-Watson Moving Average (kernel 0 - gaussian)
        33: Nadaray-Watson Moving Average (kernel 1 - epanechnikov)
        34: Nadaray-Watson Moving Average (kernel 2 - rectangular)
        35: Nadaray-Watson Moving Average (kernel 3 - triangular)
        36: Nadaray-Watson Moving Average (kernel 4 - biweight)
        37: Nadaray-Watson Moving Average (kernel 5 - cosine)
    - ma_period (int): Number of periods for the Moving Average calculation.

    Returns:
    - np.ndarray: Numpy array containing the calculated Moving Average values.
    """
    # print(f'{np_df} {type} {MA_period}')
    ma_types = {
        0: lambda ohlcv_array, period: talib.SMA(ohlcv_array[:, 3], timeperiod=period),
        1: lambda ohlcv_array, period: talib.EMA(ohlcv_array[:, 3], timeperiod=period),
        2: lambda ohlcv_array, period: talib.WMA(ohlcv_array[:, 3], timeperiod=period),
        3: lambda ohlcv_array, period: talib.KAMA(ohlcv_array[:, 3], timeperiod=period),
        4: lambda ohlcv_array, period: talib.TRIMA(
            ohlcv_array[:, 3], timeperiod=period
        ),
        5: lambda ohlcv_array, period: talib.DEMA(ohlcv_array[:, 3], timeperiod=period),
        6: lambda ohlcv_array, period: talib.TEMA(ohlcv_array[:, 3], timeperiod=period),
        7: lambda ohlcv_array, period: talib.T3(ohlcv_array[:, 3], timeperiod=period),
        8: lambda ohlcv_array, period: talib.MAMA(ohlcv_array[:, 3])[0],
        9: lambda ohlcv_array, period: talib.LINEARREG(
            ohlcv_array[:, 3], timeperiod=period
        ),
        10: lambda ohlcv_array, period: finTA.SMM(
            pd.DataFrame(
                ohlcv_array[:, :5], columns=["open", "high", "low", "close", "volume"]
            ),
            period,
        ).to_numpy(),
        11: lambda ohlcv_array, period: finTA.SSMA(
            pd.DataFrame(
                ohlcv_array[:, :5], columns=["open", "high", "low", "close", "volume"]
            ),
            period,
        ).to_numpy(),
        12: lambda ohlcv_array, period: finTA.VAMA(
            pd.DataFrame(
                ohlcv_array[:, :5], columns=["open", "high", "low", "close", "volume"]
            ),
            period,
        ).to_numpy(),
        13: lambda ohlcv_array, period: finTA.ZLEMA(
            pd.DataFrame(
                ohlcv_array[:, :5], columns=["open", "high", "low", "close", "volume"]
            ),
            max(4, period),
        ).to_numpy(),
        # One of Top3 slowest MAs to calculate:
        14: lambda ohlcv_array, period: finTA.EVWMA(
            pd.DataFrame(
                ohlcv_array[:, :5], columns=["open", "high", "low", "close", "volume"]
            ),
            period,
        ).to_numpy(),
        15: lambda ohlcv_array, period: finTA.SMMA(
            pd.DataFrame(
                ohlcv_array[:, :5], columns=["open", "high", "low", "close", "volume"]
            ),
            period,
        ).to_numpy(),
        16: lambda ohlcv_array, period: p_ta.vwma(
            pd.Series(ohlcv_array[:, 3]), pd.Series(ohlcv_array[:, 4]), length=period
        ).to_numpy(),
        # One of Top3 slowest MAs to calculate:
        17: lambda ohlcv_array, period: p_ta.swma(
            pd.Series(ohlcv_array[:, 3]), length=period, asc=True
        ).to_numpy(),
        # One of Top3 slowest MAs to calculate:
        18: lambda ohlcv_array, period: p_ta.swma(
            pd.Series(ohlcv_array[:, 3]), length=period, asc=False
        ).to_numpy(),
        19: lambda ohlcv_array, period: ti.ehma(ohlcv_array[:, 3], period),
        20: lambda ohlcv_array, period: ti.lma(ohlcv_array[:, 3], period),
        21: lambda ohlcv_array, period: ti.shmma(ohlcv_array[:, 3], period),
        22: lambda ohlcv_array, period: ti.ahma(ohlcv_array[:, 3], period),
        23: lambda ohlcv_array, period: HullMA(ohlcv_array[:, 3], max(period, 4)),
        24: lambda ohlcv_array, period: VWMA(
            ohlcv_array[:, 3], ohlcv_array[:, 4], timeperiod=period
        ),
        25: lambda ohlcv_array, period: RMA(ohlcv_array[:, 3], timeperiod=period),
        26: lambda ohlcv_array, period: ALMA(ohlcv_array[:, 3], timeperiod=period),
        27: lambda ohlcv_array, period: HammingMA(ohlcv_array[:, 3], period),
        28: lambda ohlcv_array, period: LWMA(ohlcv_array[:, 3], period),
        29: lambda ohlcv_array, period: MGD(ohlcv_array[:, 3], period),
        30: lambda ohlcv_array, period: GMA(ohlcv_array[:, 3], period),
        31: lambda ohlcv_array, period: FBA(ohlcv_array[:, 3], period),
        # 32: lambda ohlcv_array, period: FWMA(ohlcv_array[:, 3], period),
        32: lambda ohlcv_array, period: NadarayWatsonMA(
            ohlcv_array[:, 3], period, kernel=0
        ),
        33: lambda ohlcv_array, period: NadarayWatsonMA(
            ohlcv_array[:, 3], period, kernel=1
        ),
        34: lambda ohlcv_array, period: NadarayWatsonMA(
            ohlcv_array[:, 3], period, kernel=2
        ),
        35: lambda ohlcv_array, period: NadarayWatsonMA(
            ohlcv_array[:, 3], period, kernel=3
        ),
        36: lambda ohlcv_array, period: NadarayWatsonMA(
            ohlcv_array[:, 3], period, kernel=4
        ),
        37: lambda ohlcv_array, period: NadarayWatsonMA(
            ohlcv_array[:, 3], period, kernel=5
        ),
    }
    # 22: lambda np_df,period: VAMA(np_df[:,3], np_df[:,4], period),
    # 31: lambda np_df,period: VIDYA(np_df[:,3], talib.CMO(np_df[:,3], period), period)
    return ma_types[ma_type](np_df.astype(np.float64), ma_period)


def get_1D_MA(close: np.ndarray, ma_type: int, ma_period: int) -> np.ndarray:
    """
    Calculate the Moving Average (MA) of a given type for a specified period.

    Parameters:
    - close (np.ndarray): Array of closing prices.
    - ma_type (int): Type of Moving Average:
        0: Simple Moving Average (SMA)
        1: Exponential Moving Average (EMA)
        2: Weighted Moving Average (WMA)
        3: Kaufman's Adaptive Moving Average (KAMA)
        4: Triangular Moving Average (TRIMA)
        5: Double Exponential Moving Average (DEMA)
        6: Triple Exponential Moving Average (TEMA)
        7: Triple Exponential Moving Average (T3)
        8: MESA Adaptive Moving Average (MAMA)
        9: Linear Regression Moving Average (LINEARREG)
        10: Exponential Hull Moving Average (EHMA)
        11: Laguerre Moving Average (LMA)
        12: Symmetric Hull Moving Average (SHMMA)
        13: Asymmetric Hull Moving Average (AHMA)
        14: Running Moving Average (RMA)
        15: Hull Moving Average (HullMA)
        16: Arnaud Legoux Moving Average (ALMA)
        17: Hamming Moving Average (HammingMA)
        18: Linear Weighted Moving Average (LWMA)
        #19: Geometric Moving Average (GMA)
        20: Fractal Adaptive Moving Average (FBA)
        21-26: Nadaray-Watson Moving Average with various kernels (0-5)
    - ma_period (int): Number of periods to consider for the Moving Average.

    Returns:
    - np.ndarray: Calculated Moving Average values corresponding to the input parameters.
    """
    ma_types = {
        0: lambda c, p: talib.SMA(c, timeperiod=p),
        1: lambda c, p: talib.EMA(c, timeperiod=p),
        2: lambda c, p: talib.WMA(c, timeperiod=p),
        3: lambda c, p: talib.KAMA(c, timeperiod=p),
        4: lambda c, p: talib.TRIMA(c, timeperiod=p),
        5: lambda c, p: talib.DEMA(c, timeperiod=p),
        6: lambda c, p: talib.TEMA(c, timeperiod=p),
        7: lambda c, p: talib.T3(c, timeperiod=p),
        8: lambda c, p: talib.MAMA(c)[0],
        9: lambda c, p: talib.LINEARREG(c, timeperiod=p),
        10: lambda c, p: ti.ehma(c, p),
        11: lambda c, p: ti.lma(c, p),
        12: lambda c, p: ti.shmma(c, p),
        13: lambda c, p: ti.ahma(c, p),
        14: lambda c, p: RMA(c, timeperiod=p),
        15: lambda c, p: HullMA(c, max(p, 4)),
        16: lambda c, p: ALMA(c, timeperiod=p),
        17: lambda c, p: HammingMA(c, p),
        18: lambda c, p: LWMA(c, p),
        # 19: lambda c, p: GMA(c, p),
        19: lambda c, p: FBA(c, p),
        # 21: lambda c, p: FWMA(c, p),
        20: lambda c, p: NadarayWatsonMA(c, p, kernel=0),
        21: lambda c, p: NadarayWatsonMA(c, p, kernel=1),
        22: lambda c, p: NadarayWatsonMA(c, p, kernel=2),
        23: lambda c, p: NadarayWatsonMA(c, p, kernel=3),
        24: lambda c, p: NadarayWatsonMA(c, p, kernel=4),
        25: lambda c, p: NadarayWatsonMA(c, p, kernel=5),
    }
    # 22: lambda np_df,period: VAMA(np_df[:,3], np_df[:,4], period),
    # 31: lambda np_df,period: VIDYA(np_df[:,3], talib.CMO(np_df[:,3], period), period)
    return np.nan_to_num(ma_types[ma_type](close, int(ma_period)))


def get_ma_from_source(
        np_df: np.ndarray, ma_type: int, ma_period: int, source: str = "close"
) -> np.ndarray:
    column_map = {
        "open": np_df[:, 0],
        "high": np_df[:, 1],
        "low": np_df[:, 2],
        "close": np_df[:, 3],
        "hl2": (np_df[:, 1] + np_df[:, 2]) / 2,
        "hlc3": (np_df[:, 1] + np_df[:, 2] + np_df[:, 3]) / 3,
        "ohlc4": (np_df[:, 0] + np_df[:, 1] + np_df[:, 2] + np_df[:, 3]) / 4,
        "hlcc4": (np_df[:, 1] + np_df[:, 2] + np_df[:, 3] + np_df[:, 3]) / 4,
        "vwap": np_df[:, 5],
    }
    try:
        s = column_map[source]
    except KeyError:
        raise ValueError(f"Unknown MA source column: {source}")

    ma_types = {
        0: lambda ohlcv_array, period: talib.SMA(s, timeperiod=period),
        1: lambda ohlcv_array, period: talib.EMA(s, timeperiod=period),
        2: lambda ohlcv_array, period: talib.WMA(s, timeperiod=period),
        3: lambda ohlcv_array, period: talib.KAMA(s, timeperiod=period),
        4: lambda ohlcv_array, period: talib.TRIMA(s, timeperiod=period),
        5: lambda ohlcv_array, period: talib.DEMA(s, timeperiod=period),
        6: lambda ohlcv_array, period: talib.TEMA(s, timeperiod=period),
        7: lambda ohlcv_array, period: talib.T3(s, timeperiod=period),
        8: lambda ohlcv_array, period: talib.MAMA(s)[0],
        9: lambda ohlcv_array, period: talib.LINEARREG(s, timeperiod=period),
        10: lambda ohlcv_array, period: p_ta.vwma(
            pd.Series(s), pd.Series(ohlcv_array[:, 4]), length=period
        ).to_numpy(),
        # One of Top3 slowest MAs to calculate:
        11: lambda ohlcv_array, period: p_ta.swma(
            pd.Series(s), length=period, asc=True
        ).to_numpy(),
        # One of Top3 slowest MAs to calculate:
        12: lambda ohlcv_array, period: p_ta.swma(
            pd.Series(s), length=period, asc=False
        ).to_numpy(),
        13: lambda ohlcv_array, period: ti.ehma(s, period),
        14: lambda ohlcv_array, period: ti.lma(s, period),
        15: lambda ohlcv_array, period: ti.shmma(s, period),
        16: lambda ohlcv_array, period: ti.ahma(s, period),
        17: lambda ohlcv_array, period: HullMA(s, max(period, 4)),
        18: lambda ohlcv_array, period: VWMA(s, ohlcv_array[:, 4], timeperiod=period),
        19: lambda ohlcv_array, period: RMA(s, timeperiod=period),
        20: lambda ohlcv_array, period: ALMA(s, timeperiod=period),
        21: lambda ohlcv_array, period: HammingMA(s, period),
        22: lambda ohlcv_array, period: LWMA(s, period),
        23: lambda ohlcv_array, period: MGD(s, period),
        24: lambda ohlcv_array, period: GMA(s, period),
        25: lambda ohlcv_array, period: FBA(s, period),
        26: lambda ohlcv_array, period: NadarayWatsonMA(s, period, kernel=0),
        27: lambda ohlcv_array, period: NadarayWatsonMA(s, period, kernel=1),
        28: lambda ohlcv_array, period: NadarayWatsonMA(s, period, kernel=2),
        29: lambda ohlcv_array, period: NadarayWatsonMA(s, period, kernel=3),
        30: lambda ohlcv_array, period: NadarayWatsonMA(s, period, kernel=4),
        31: lambda ohlcv_array, period: NadarayWatsonMA(s, period, kernel=5),
    }
    # 22: lambda np_df,period: VAMA(np_df[:,3], np_df[:,4], period),
    # 31: lambda np_df,period: VIDYA(np_df[:,3], talib.CMO(np_df[:,3], period), period)
    return np.nan_to_num(ma_types[ma_type](np_df, int(ma_period)))


def any_ma_sig(np_close, np_xMA, np_ATR, atr_multi=1.0):
    return np.where(np_ATR != 0, (np_xMA - np_close) / (np_ATR * atr_multi), 0)


# @feature_timeit
def other_features(df: pd.DataFrame, suffix=""):
    _, O, H, L, C, V, *_ = [df[col].to_numpy() for col in df.columns]
    df[f"RSI14{suffix}"] = talib.RSI(C, timeperiod=14)
    df[f"RSI7{suffix}"] = talib.RSI(C, timeperiod=7)
    df[f"RSI3{suffix}"] = talib.RSI(C, timeperiod=3)
    df[f"ULT{suffix}"] = talib.ULTOSC(
        H, L, C, timeperiod1=7, timeperiod2=14, timeperiod3=28
    )
    df[f"ADX{suffix}"] = talib.ADX(H, L, C, timeperiod=14)
    df[f"-DI{suffix}"] = talib.MINUS_DI(H, L, C, timeperiod=14)
    df[f"+DI{suffix}"] = talib.PLUS_DI(H, L, C, timeperiod=14)
    df[f"MFI{suffix}"] = talib.MFI(H, L, C, V, timeperiod=14)
    df[f"macd{suffix}"], df[f"macdsignal{suffix}"], df[f"macdhist{suffix}"] = (
        talib.MACD(C, fastperiod=12, slowperiod=26, signalperiod=9)
    )
    df[f"ATR{suffix}"] = talib.ATR(H, L, C, timeperiod=14)
    df[f"ADOSC{suffix}"] = talib.ADOSC(H, L, C, V, fastperiod=3, slowperiod=10)
    df[f"APO{suffix}"] = talib.APO(C, fastperiod=12, slowperiod=26, matype=0)
    df[f"AROONOSC{suffix}"] = talib.AROONOSC(H, L, timeperiod=14)
    df[f"STOCHRSIfastk{suffix}"], df[f"STOCHRSIfastd{suffix}"] = talib.STOCHRSI(
        C, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
    )
    df[f"CMO{suffix}"] = talib.CMO(C, timeperiod=14)
    df[f"BOP{suffix}"] = talib.BOP(O, H, L, C)
    df[f"TRANGE{suffix}"] = talib.TRANGE(H, L, C)
    df[f"PPO{suffix}"] = talib.PPO(C, fastperiod=12, slowperiod=26, matype=0)
    df[f"WILLR{suffix}"] = talib.WILLR(H, L, C, timeperiod=14)
    df[f"KST{suffix}"] = ta.trend.kst_sig(df["Close"])
    df[f"Vortex{suffix}"] = ta.trend.VortexIndicator(
        df["High"], df["Low"], df["Close"]
    ).vortex_indicator_diff()
    df[f"STC{suffix}"] = ta.trend.STCIndicator(df["Close"]).stc()
    df[f"PVO{suffix}"] = ta.momentum.PercentageVolumeOscillator(df["Volume"]).pvo()
    df[f"AO{suffix}"] = ta.momentum.AwesomeOscillatorIndicator(
        df["High"], df["Low"]
    ).awesome_oscillator()
    bb_upper, bb_mid, bb_lower = talib.BBANDS(
        C, timeperiod=11, nbdevup=1.5, nbdevdn=1.5, matype=0
    )
    df[f"BB11-1.5_signal{suffix}"] = BB_signal(C, bb_upper, bb_mid, bb_lower)
    ### Signals
    df[f"RSI3_sig{suffix}"] = RSI_like_signal(df[f"RSI3{suffix}"], timeperiod=3)
    df[f"RSI7_sig{suffix}"] = RSI_like_signal(df[f"RSI7{suffix}"], timeperiod=7)
    df[f"RSI14_sig{suffix}"] = RSI_like_signal(df[f"RSI14{suffix}"], timeperiod=14)
    df[f"ULT_sig{suffix}"] = RSI_like_signal(df[f"ULT{suffix}"], timeperiod=28)
    df[f"ADX_sig{suffix}"] = ADX_signal(
        df[f"ADX{suffix}"], df[f"-DI{suffix}"], df[f"+DI{suffix}"]
    )
    df[f"ADX_trend_signal{suffix}"] = ADX_trend_signal(df[f"ADX{suffix}"])
    df[f"MFI_sig{suffix}"] = RSI_like_signal(df[f"MFI{suffix}"])
    df[f"MACD_cross_signal{suffix}"] = MACD_cross_signal(
        df[f"macd{suffix}"], df[f"macdsignal{suffix}"]
    )
    df[f"MACDhist_reversal_signal{suffix}"] = MACD_histogram_reversal_signal(
        df[f"macdhist{suffix}"]
    )
    df[f"MACD_zerocross_signal{suffix}"] = MACD_zero_cross_signal(
        df[f"macd{suffix}"], df[f"macdsignal{suffix}"]
    )
    df[f"BB_signal{suffix}"] = BB_signal(
        df[f"mid{suffix}"].to_numpy(),
        df[f"up_x1"].to_numpy(),
        df["low_x1"].to_numpy(),
        df["up_x2"].to_numpy(),
        df["low_x2"].to_numpy(),
        df["up_x3"].to_numpy(),
        df["low_x3"].to_numpy(),
        C,
    )
    return df


def signal_features(df: pd.DataFrame, suffix: str = ""):
    _, O, H, L, C, V, *_ = [df[col].to_numpy() for col in df.columns]
    ### Signals
    df[f"RSI3_signal{suffix}"] = RSI_like_signal(talib.RSI(C, 3), 3)
    df[f"RSI7_signal{suffix}"] = RSI_like_signal(talib.RSI(C, 7), 7)
    df[f"RSI14_signal{suffix}"] = RSI_like_signal(talib.RSI(C, 14), 14)
    df[f"ULT7-14-28_signal{suffix}"] = RSI_like_signal(
        talib.ULTOSC(H, L, C, 7, 14, 28), 28
    )
    df[f"ADX14_signal{suffix}"] = ADX_signal(
        talib.ADX(H, L, C, 14), talib.MINUS_DI(H, L, C, 14), talib.PLUS_DI(H, L, C, 14)
    )
    df[f"ADX14_trend_signal{suffix}"] = ADX_trend_signal(talib.ADX(H, L, C, 14))
    df[f"MFI14_signal{suffix}"] = RSI_like_signal(talib.MFI(H, L, C, V, 14), 14)
    macd, macd_sig, macd_hist = talib.MACD(C, 12, 26, 9)
    df[f"MACD12-26-9_cross_signal{suffix}"] = MACD_cross_signal(macd, macd_sig)
    df[f"MACD12-26-9hist_reversal_signal{suffix}"] = MACD_histogram_reversal_signal(
        macd_hist
    )
    df[f"MACD12-26-9zerocross_signal{suffix}"] = MACD_zero_cross_signal(macd, macd_sig)
    bb_upper, bb_mid, bb_lower = talib.BBANDS(
        C, timeperiod=11, nbdevup=1.5, nbdevdn=1.5, matype=0
    )
    df[f"BB11-1.5_signal{suffix}"] = BB_signal(C, bb_upper, bb_mid, bb_lower)
    df["move_probablity"] = move_probability(O, C)
    df["SUM_OF_SIGNALS"] = df.iloc[:, 6:].sum(axis=1)
    return df


def signal_features_periods(
        df: pd.DataFrame, periods: list[int], suffix: str = ""
) -> pd.DataFrame:
    print(f"Periods used: {periods}")
    _, O, H, L, C, V, *_ = [df[col].to_numpy() for col in df.columns]
    for f in periods:
        _f = int((2 / 3) * f)
        ff, fff = 2 * f, 3 * f
        ### Signals
        df[f"RSI{f}_signal{suffix}"] = RSI_like_signal(talib.RSI(C, f), f)
        df[f"MFI{f}_signal{suffix}"] = RSI_like_signal(talib.MFI(H, L, C, V, f), f)
        df[f"ULT{f}-{ff}-{3 * f}_signal{suffix}"] = RSI_like_signal(
            talib.ULTOSC(H, L, C, f, ff, fff), fff
        )
        adx, mDI, pDI = (
            talib.ADX(H, L, C, f),
            talib.MINUS_DI(H, L, C, f),
            talib.PLUS_DI(H, L, C, f),
        )
        df[f"ADX{f}_signal{suffix}"] = ADX_signal(adx, mDI, pDI)
        df[f"ADX{f}trend_signal{suffix}"] = ADX_trend_signal(adx, mDI, pDI)
        for i in range(0, 9):
            macd, macd_signal, macd_hist = talib.MACDEXT(
                C,
                fastperiod=f,
                slowperiod=ff,
                signalperiod=_f,
                fastmatype=i,
                slowmatype=i,
                signalmatype=i,
            )
            df[f"{i}MACD{f}-{ff}-{_f}_cross_signal{suffix}"] = MACD_cross_signal(
                macd, macd_signal
            )
            df[f"{i}MACD{f}-{ff}-{_f}hist_reversal_signal{suffix}"] = (
                MACD_histogram_reversal_signal(macd_hist)
            )
            df[f"{i}MACD{f}-{ff}-{_f}zerocross_signal{suffix}"] = (
                MACD_zero_cross_signal(macd, macd_signal)
            )
        bb_upper, bb_mid, bb_lower = talib.BBANDS(
            C, timeperiod=f, nbdevup=1.5, nbdevdn=1.5, matype=0
        )
        df[f"BB{f}-{1.5}_signal{suffix}"] = BB_signal(C, bb_upper, bb_mid, bb_lower)
        # df['move_probablity'] = move_probability(O, C)
    df["SUM_OF_SIGNALS"] = df.iloc[:, 6:].sum(axis=1)
    return df


def simple_rl_features(ohlcv_df: pd.DataFrame, suffix: str = ""):
    _, O, H, L, C, V, *_ = [ohlcv_df[col].to_numpy() for col in ohlcv_df.columns]
    ohlcv_df[f"ADX14{suffix}"] = talib.ADX(H, L, C, 14)
    ohlcv_df[f"ADXR14{suffix}"] = talib.ADXR(H, L, C, 14)
    ohlcv_df[f"APO12-26{suffix}"] = talib.APO(C, fastperiod=12, slowperiod=26, matype=0)
    ohlcv_df[f"AROON14_DOWN{suffix}"], ohlcv_df[f"AROON14_UP{suffix}"] = talib.AROON(
        H, L, 14
    )
    ohlcv_df[f"AROONOSC14{suffix}"] = talib.AROONOSC(H, L, 14)
    ohlcv_df[f"BOP{suffix}"] = talib.BOP(O, H, L, C)
    ohlcv_df[f"CCI14{suffix}"] = talib.CCI(H, L, C, 14)
    ohlcv_df[f"CMO14{suffix}"] = talib.CMO(C, 14)
    ohlcv_df[f"DX14{suffix}"] = talib.DX(H, L, C, 14)
    (
        ohlcv_df[f"MACD12-26{suffix}"],
        ohlcv_df[f"MACDsignal9{suffix}"],
        ohlcv_df[f"MACD12-26hist{suffix}"],
    ) = talib.MACD(C, 12, 26, 9)
    ohlcv_df[f"MFI14{suffix}"] = talib.MFI(H, L, C, V, 14)
    ohlcv_df[f"MINUS_DI14{suffix}"] = talib.MINUS_DI(H, L, C, 14)
    ohlcv_df[f"MINUS_DM14{suffix}"] = talib.MINUS_DM(H, L, 14)
    ohlcv_df[f"PLUS_DI14{suffix}"] = talib.PLUS_DI(H, L, C, 14)
    ohlcv_df[f"PLUS_DM14{suffix}"] = talib.PLUS_DM(H, L, 14)
    ohlcv_df[f"MOM10{suffix}"] = talib.MOM(C, 10)
    ohlcv_df[f"PPO12-26{suffix}"] = talib.PPO(C, fastperiod=12, slowperiod=26, matype=0)
    ohlcv_df[f"ROC10{suffix}"] = talib.ROC(C, 10)
    ohlcv_df[f"ROCP10{suffix}"] = talib.ROCP(C, 10)
    ohlcv_df[f"ROCR100-10{suffix}"] = talib.ROCR100(C, 10)
    ohlcv_df[f"RSI14{suffix}"] = talib.RSI(C, 14)
    ohlcv_df[f"SLOWK3{suffix}"], ohlcv_df[f"SLOWD3{suffix}"] = talib.STOCH(
        H,
        L,
        C,
        fastk_period=5,
        slowk_period=3,
        slowk_matype=0,
        slowd_period=3,
        slowd_matype=0,
    )
    ohlcv_df[f"FASTK5{suffix}"], ohlcv_df[f"FASTD5{suffix}"] = talib.STOCHF(
        H, L, C, fastk_period=5, fastd_period=3, fastd_matype=0
    )
    ohlcv_df[f"ULTOSC7-14-28{suffix}"] = talib.ULTOSC(H, L, C, 7, 14, 28)
    ohlcv_df[f"WILLR14{suffix}"] = talib.WILLR(H, L, C, 14)
    # ohlcv_df[f'AD{suffix}'] = talib.AD(H, L, C, V)
    ohlcv_df[f"ADOSC{suffix}"] = talib.ADOSC(H, L, C, V, fastperiod=3, slowperiod=10)
    # ohlcv_df[f'OBV{suffix}'] = talib.OBV(C, V)
    ohlcv_df[f"ATR14{suffix}"] = talib.ATR(H, L, C, 14)
    ohlcv_df[f"NATR14{suffix}"] = talib.NATR(H, L, C, 14)
    ohlcv_df[f"TRANGE{suffix}"] = talib.TRANGE(H, L, C)
    # ohlcv_df[f'AVGPRICE{suffix}'] = talib.AVGPRICE(O, H, L, C)
    # ohlcv_df[f'MEDPRICE{suffix}'] = talib.MEDPRICE(H, L)
    # ohlcv_df[f'TYPPRICE{suffix}'] = talib.TYPPRICE(H, L, C)
    # ohlcv_df[f'WCLPRICE{suffix}'] = talib.WCLPRICE(H, L, C)
    return ohlcv_df


def simple_rl_features_periods(
        df: pd.DataFrame,
        periods: list[int],
        zscore_standardization: bool = False,
        suffix: str = "",
):
    print(f"Periods used: {periods}")
    _, O, H, L, C, V, *_ = [df[col].to_numpy() for col in df.columns]
    for f in periods:
        _f = int((2 / 3) * f)
        ff, fff = 2 * f, 3 * f
        df[f"ADX{f}{suffix}"] = talib.ADX(H, L, C, f)
        df[f"ADXR{f}{suffix}"] = talib.ADXR(H, L, C, f)
        df[f"AROON_DOWN{f}{suffix}"], df[f"AROON_UP{f}{suffix}"] = talib.AROON(H, L, f)
        df[f"AROONOSC{f}{suffix}"] = talib.AROONOSC(H, L, f)
        df[f"CCI{f}{suffix}"] = talib.CCI(H, L, C, f)
        df[f"CMO{f}{suffix}"] = talib.CMO(C, f)
        df[f"DX{f}{suffix}"] = talib.DX(H, L, C, f)
        # Using every talib MAtype possible (0,8]
        for i in range(0, 9):
            df[f"APO{f}-{ff}{suffix}"] = talib.APO(
                C, fastperiod=f, slowperiod=ff, matype=i
            )
            df[f"PPO{f}-{ff}{suffix}"] = talib.PPO(
                C, fastperiod=f, slowperiod=ff, matype=i
            )
            (
                df[f"MACD{f}-{ff}{suffix}"],
                df[f"MACDsignal{_f}{suffix}"],
                df[f"MACDhist{f}-{ff}{suffix}"],
            ) = talib.MACDEXT(
                C,
                fastperiod=f,
                slowperiod=ff,
                signalperiod=_f,
                fastmatype=i,
                slowmatype=i,
                signalmatype=i,
            )
            df[f"STOCH_SLOWK{_f}-{f}{suffix}"], df[f"STOCH_SLOWD{_f}-{f}{suffix}"] = (
                talib.STOCH(
                    H,
                    L,
                    C,
                    fastk_period=f,
                    slowk_period=_f,
                    slowk_matype=i,
                    slowd_period=_f,
                    slowd_matype=i,
                )
            )
            df[f"STOCH_FASTK{_f}-{f}{suffix}"], df[f"STOCH_FASTD{_f}-{f}{suffix}"] = (
                talib.STOCHF(H, L, C, fastk_period=f, fastd_period=_f, fastd_matype=i)
            )
            df[f"STOCHRSI_FASTK{_f}-{f}-{ff}"], df[f"STOCHRSI_FASTD{_f}-{f}-{ff}"] = (
                talib.STOCHRSI(
                    C, timeperiod=ff, fastk_period=f, fastd_period=_f, fastd_matype=i
                )
            )
        df[f"MFI{f}{suffix}"] = talib.MFI(H, L, C, V, f)
        df[f"MINUS_DI{f}{suffix}"] = talib.MINUS_DI(H, L, C, f)
        df[f"MINUS_DM{f}{suffix}"] = np.emath.log(talib.MINUS_DM(H, L, f))
        df[f"PLUS_DI{f}{suffix}"] = talib.PLUS_DI(H, L, C, f)
        df[f"PLUS_DM{f}{suffix}"] = np.emath.log(talib.PLUS_DM(H, L, f))
        df[f"MOM{f}{suffix}"] = talib.MOM(C, f)
        df[f"ROC{f}{suffix}"] = talib.ROC(C, f)
        df[f"ROCP{f}{suffix}"] = talib.ROCP(C, f)
        df[f"ROCR100-{f}{suffix}"] = talib.ROCR100(C, f)
        df[f"RSI{f}{suffix}"] = talib.RSI(C, f)
        df[f"ULTOSC{f}-{ff}-{fff}{suffix}"] = talib.ULTOSC(H, L, C, f, ff, fff)
        df[f"WILLR{f}{suffix}"] = talib.WILLR(H, L, C, f)
        df[f"ADOSC{f}-{_f}{suffix}"] = talib.ADOSC(
            H, L, C, V, fastperiod=_f, slowperiod=f
        )
        df[f"ATR{f}{suffix}"] = talib.ATR(H, L, C, f)
        df[f"NATR{f}{suffix}"] = talib.NATR(H, L, C, f)
    df[f"BOP{suffix}"] = talib.BOP(O, H, L, C)
    df[f"OBV{suffix}"] = talib.OBV(C, V)
    df[f"TRANGE{suffix}"] = talib.TRANGE(H, L, C)
    df = custom_features(df)
    # To avoid NaN indicator values at the beginning
    # longest_period = max(periods) * 25
    # df = df.tail(len(df) - longest_period)
    if not zscore_standardization:
        print(df.columns[df.isna().any()].tolist())
        return df
    else:
        cols_to_exclude = ["Opened", "Open", "High", "Low", "Close"]
        cols_to_zscore = [col for col in df.columns if col not in cols_to_exclude]
        df[cols_to_zscore] = df[cols_to_zscore].apply(
            lambda x: (x - np.mean(x)) / np.std(x)
        )
        print(df.columns[df.isna().any()].tolist())
        return df


def custom_features(df, suffix=""):
    _, O, H, L, C, *_ = [df[col].to_numpy() for col in df.columns]
    # OHLC simple features
    df[f"candle_size{suffix}"] = H - L
    df[f"candle_body_size{suffix}"] = np.where(
        C > O,
        (C - O) / df[f"candle_size{suffix}"],
        (O - C) / df[f"candle_size{suffix}"],
    )
    df[f"candle_upper_wick{suffix}"] = np.where(
        C > O,
        (H - C) / df[f"candle_size{suffix}"],
        (H - O) / df[f"candle_size{suffix}"],
    )
    df[f"candle_lower_wick{suffix}"] = np.where(
        C > O,
        (O - L) / df[f"candle_size{suffix}"],
        (C - L) / df[f"candle_size{suffix}"],
    )
    df[f"hourly_seasonality{suffix}"] = hourly_seasonality(df)
    df[f"daily_seasonality{suffix}"] = daily_seasonality(df)
    # df['volume_probability{suffix}'] = volume_probability(V)
    df[f"move_probability{suffix}"] = move_probability(O, C)
    df[f"price_levels{suffix}"] = price_levels(O, C)
    return df


def blank_features(df, *args, **kwargs):
    return df


TA_FEATURES_TEMPLATE = {
    "None": blank_features,
    "custom": custom_features,
    "signals": signal_features,
}


def get_combined_intervals_df(
        ticker, interval_list, market_type="um", data_type="klines", template="None"
):
    df = get_data.by_BinanceVision(
        ticker=ticker,
        interval=interval_list[0],
        market_type=market_type,
        data_type=data_type,
    )
    df = TA_FEATURES_TEMPLATE[template](df, ticker)
    for itv in interval_list[1:]:
        _df = get_data.by_BinanceVision(
            ticker=ticker, interval=itv, market_type=market_type, data_type=data_type
        )
        _df["Opened"] = _df["Opened"] + (
                _df.iloc[-1]["Opened"] - _df.iloc[-2]["Opened"]
        )
        _df = TA_FEATURES_TEMPLATE[template](_df, ticker, suffix="_" + itv)
        df = pd.merge_asof(
            df, _df, on="Opened", direction="backward", suffixes=("", "_" + itv)
        )
    df.fillna(method="ffill", inplace=True)
    return df
