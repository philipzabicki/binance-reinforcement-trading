from typing import Any

import numpy as np
import talib
from numba import jit
from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit


### Helper Functions ###
@jit(nopython=True, nogil=True, cache=True)
def fib_to(n, normalization=True):
    fibs = np.empty(n)
    fibs[0], fibs[1] = 1, 2
    for i in range(2, n):
        fibs[i] = fibs[i - 1] + fibs[i - 2]
    if normalization:
        return (fibs - np.min(fibs)) / (np.max(fibs) - np.min(fibs))
    else:
        return fibs


def HullMA(
    close: list | np.ndarray, timeperiod: int
) -> ndarray[Any, dtype[floating[_64Bit]]]:
    return talib.WMA(
        np.nan_to_num(talib.WMA(close, timeperiod // 2) * 2)
        - np.nan_to_num(talib.WMA(close, timeperiod)),
        int(np.sqrt(timeperiod)),
    )


# @feature_timeit
@jit(nopython=True, nogil=True, cache=True)
def RMA(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    Calculate the Relative Moving Average (RMA) of a given array of closing prices.

    Args:
        close (np.ndarray): An array of closing prices.
        timeperiod (int): The time period to consider for the RMA calculation.

    Returns:
        np.ndarray[np.float64]: An array of RMA values of the same length as the input 'close'.
                                Preceded by 0.0 times timeperiod.

    Notes:
        A Relative Moving Average adds more weight to recent data (and gives less importance to older data).
        This makes the RMA similar to the EMA, although it’s somewhat slower to respond than an EMA is.
    """
    alpha = 1.0 / timeperiod
    # rma = [0.0] * len(close)
    rma = np.zeros_like(close)
    # Calculating the SMA for the first 'length' values
    sma = np.sum(close[:timeperiod]) / timeperiod
    rma[timeperiod - 1] = sma
    # Calculating the rma for the rest of the values
    for i in range(timeperiod, len(close)):
        rma[i] = (alpha * close[i]) + ((1 - alpha) * rma[i - 1])
    return rma


@jit(nopython=True, nogil=True, cache=True)
def LSMA(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    Calculate the Least Squares Moving Average (LSMA) of a time series.

    Args:
        close (np.ndarray): An array of closing prices or time series data.
        timeperiod (int): The time period for the LSMA calculation.

    Returns:
        np.ndarray: An array containing the LSMA values of the input data.

    This function calculates the LSMA for the given input data. LSMA is a linear
    regression-based moving average, which fits a linear line to 'timeperiod'
    data points and calculates the moving average based on the slope and
    intercept of the fitted line.

    Note:
        - The input 'close' array should be a NumPy ndarray.
        - The output array will have 'np.nan' values for the first 'timeperiod - 1'
          elements since there are not enough data points to perform the calculation.

    """
    close = np.ascontiguousarray(close)
    lsma = np.empty_like(close)
    A = np.column_stack((np.arange(timeperiod), np.ones(timeperiod)))
    AT = np.ascontiguousarray(A.T)
    ATA_inv = np.linalg.inv(np.dot(AT, A))
    for i in range(timeperiod - 1, len(close)):
        m, c = np.dot(ATA_inv, np.dot(AT, close[i - timeperiod + 1 : i + 1]))
        lsma[i] = m * (timeperiod - 1) + c
    return lsma


# @feature_timeit
# @jit(nopython=True, nogil=True, cache=True)
def ALMA(
    close: np.ndarray, timeperiod: int, offset: float = 0.85, sigma: int = 6
) -> np.ndarray:
    """
    Calculate the Arnaud Legoux Moving Average (ALMA) for a given input time series.

    Args:
        close (np.ndarray): An array of closing prices.
        timeperiod (int): The number of periods to consider for the ALMA calculation.
        offset (float, optional): The offset factor for the ALMA calculation. Default is 0.85.
        sigma (int, optional): The standard deviation factor for the ALMA calculation. Default is 6.

    Returns:
        np.ndarray: An array containing the ALMA values with NaN values padded at the beginning.

    ALMA (Arnaud Legoux Moving Average) is a weighted moving average that is designed to reduce lag
    in the moving average by incorporating a Gaussian distribution. This function calculates the ALMA
    for the given input time series, with customizable parameters for the offset and sigma.

    """
    # close = np.ascontiguousarray(close)
    m = offset * (timeperiod - 1)
    s = timeperiod / sigma
    denominator = 2 * s**2
    wtd = np.array(
        [np.exp(-((i - m) ** 2) / denominator) for i in range(timeperiod)],
        dtype=close.dtype,
    )
    wtd /= wtd.sum()
    alma = np.convolve(close, wtd, "valid")
    return np.concatenate((np.zeros(timeperiod - 1), alma))


# @jit(nopython=True, nogil=True, cache=True)
def GMA(close: np.ndarray, period: int) -> np.ndarray[np.float64]:
    close = np.absolute(close)
    gma = np.zeros_like(close)
    log_close = np.log(close)
    exponent = 1 / period
    window_sum = np.sum(log_close[:period])
    # print(f'log_close {log_close}')
    # print(f'window_sum {window_sum}')
    gma[period - 1] = window_sum * exponent
    for i in range(period, len(close)):
        window_sum -= log_close[i - period]
        window_sum += log_close[i]
        gma[i] = window_sum * exponent
        # print(f'window_sum {window_sum}')
        # print(f'gma[i] {gma[i]}')
        # sleep(5)
    return np.exp(gma)


@jit(nopython=True, nogil=True, cache=True)
def VWMAv1(close: np.ndarray, volume: np.ndarray, timeperiod: int):
    close = np.ascontiguousarray(close)
    volume = np.ascontiguousarray(volume)
    vwma = np.array(
        [
            np.sum(close[i - timeperiod : i] * volume[i - timeperiod : i])
            / np.sum(volume[i - timeperiod : i])
            for i in range(timeperiod, len(close) + 1)
        ]
    )
    return np.concatenate((np.zeros(timeperiod - 1), vwma))


@jit(nopython=True, nogil=True, cache=True)
def VWMA(close: np.ndarray, volume: np.ndarray, timeperiod: int):
    vwma = np.zeros_like(close)
    window_sum_volume = np.sum(volume[:timeperiod])
    window_sum_cxv = np.sum(close[:timeperiod] * volume[:timeperiod])
    vwma[timeperiod - 1] = window_sum_cxv / window_sum_volume
    for i in range(timeperiod, len(close)):
        window_sum_volume -= volume[i - timeperiod]
        window_sum_volume += volume[i]
        window_sum_cxv -= close[i - timeperiod] * volume[i - timeperiod]
        window_sum_cxv += close[i] * volume[i]
        vwma[i] = window_sum_cxv / window_sum_volume
    return vwma


# @feature_timeit
# @jit(nopython=True, nogil=True, cache=True)
def HammingMA(close: np.ndarray, timeperiod: int) -> np.ndarray:
    """
    Calculate the Hamming Moving Average (HMA) of a given numpy array of closing prices.

    Args:
        close (np.ndarray): A numpy array containing the closing prices.
        timeperiod (int): The period over which to calculate the HMA.

    Returns:
        np.ndarray[np.float64]: The Hamming Moving Average of the closing prices as a numpy array.

    This function computes the HMA by applying a Hamming window to the closing prices and then
    performing a convolution. The resulting HMA is returned as a numpy array.

    The Hamming window is applied over the specified 'timeperiod', and the 'mode' is set to 'valid'.

    Note:
    - The 'close' numpy array should have at least 'timeperiod' data points.

    """
    w = np.hamming(timeperiod)
    hma = np.convolve(close, w, mode="valid") / w.sum()
    return np.concatenate((np.zeros(timeperiod - 1), hma))


# @jit(nopython=True, nogil=True, cache=True)
# def gaussian_kernel(x: np.ndarray) -> np.ndarray[np.float64]:
#     return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
#
#
# @jit(nopython=True, nogil=True, cache=True)
# def epanechnikov_kernel(x: np.ndarray) -> np.ndarray[np.float64]:
#     return np.where(np.abs(x) <= 1, 3 / 4 * (1 - x ** 2), 0)
#
#
# @jit(nopython=True, nogil=True, cache=True)
# def rectangular_kernel(x: np.ndarray) -> np.ndarray[np.float64]:
#     return np.where(np.abs(x) <= 1, 0.5, 0)
#
#
# @jit(nopython=True, nogil=True, cache=True)
# def triangular_kernel(x: np.ndarray) -> np.ndarray[np.float64]:
#     return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)
#
#
# @jit(nopython=True, nogil=True, cache=True)
# def biweight_kernel(x: np.ndarray) -> np.ndarray[np.float64]:
#     return np.where(np.abs(x) <= 1, (15 / 16) * (1 - x ** 2) ** 2, 0)
#
#
# @jit(nopython=True, nogil=True, cache=True)
# def cosine_kernel(x: np.ndarray) -> np.ndarray[np.float64]:
#     return np.where(np.abs(x) <= 1, np.pi / 4 * np.cos(np.pi / 2 * x), 0)
#
#
# @jit(nopython=True, nogil=True, cache=True)
# def NadarayWatsonMA(close: np.ndarray, timeperiod: int, kernel: int = 0) -> np.ndarray[np.float64]:
#     # nwma = np.empty_like(close)
#     if timeperiod % 2 == 1:
#         distances = np.concatenate((np.arange(timeperiod // 2 + 1, 0, -1, dtype=close.dtype),
#                                     np.arange(2, timeperiod // 2 + 2, dtype=close.dtype)))
#     else:
#         distances = np.concatenate((np.arange(timeperiod // 2, 0, -1, dtype=close.dtype),
#                                     np.arange(1, timeperiod // 2 + 1, dtype=close.dtype)))
#     if kernel == 0:
#         weights = gaussian_kernel(distances / timeperiod)
#     elif kernel == 1:
#         weights = epanechnikov_kernel(distances / timeperiod)
#     elif kernel == 2:
#         weights = rectangular_kernel(distances / timeperiod)
#     elif kernel == 3:
#         weights = triangular_kernel(distances / timeperiod)
#     elif kernel == 4:
#         weights = biweight_kernel(distances / timeperiod)
#     elif kernel == 5:
#         weights = cosine_kernel(distances / timeperiod)
#     else:
#         raise ValueError("kernel argument must be int from 0 to 5")
#     weights_sum = weights.sum()
#     nwma = np.array([(weights @ close[i - timeperiod + 1:i + 1]) / weights_sum for i in range(timeperiod - 1, len(close))])
#     return np.concatenate((np.zeros(timeperiod-1), nwma))


@jit(nopython=True, nogil=True, cache=True)
def NadarayWatsonMA(close: np.ndarray, timeperiod: int, kernel: int = 0) -> np.ndarray:
    close = np.ascontiguousarray(close)
    # nwma = np.empty_like(close)
    if timeperiod % 2 == 1:
        distances = np.concatenate(
            (np.arange(timeperiod // 2 + 1, 0, -1), np.arange(2, timeperiod // 2 + 2))
        )
    else:
        distances = np.concatenate(
            (np.arange(timeperiod // 2, 0, -1), np.arange(1, timeperiod // 2 + 1))
        )
    if kernel == 0:
        weights = np.ascontiguousarray(
            np.exp(-0.5 * (distances / timeperiod) ** 2) / np.sqrt(2 * np.pi)
        )
    elif kernel == 1:
        weights = np.ascontiguousarray(
            np.where(
                np.abs(distances / timeperiod) <= 1,
                3 / 4 * (1 - (distances / timeperiod) ** 2),
                0,
            )
        )
    elif kernel == 2:
        weights = np.ascontiguousarray(
            np.where(np.abs(distances / timeperiod) <= 1, 0.5, 0)
        )
    elif kernel == 3:
        weights = np.ascontiguousarray(
            np.where(
                np.abs(distances / timeperiod) <= 1,
                1 - np.abs(distances / timeperiod),
                0,
            )
        )
    elif kernel == 4:
        weights = np.ascontiguousarray(
            np.where(
                np.abs(distances / timeperiod) <= 1,
                (15 / 16) * (1 - (distances / timeperiod) ** 2) ** 2,
                0,
            )
        )
    elif kernel == 5:
        weights = np.ascontiguousarray(
            np.where(
                np.abs(distances / timeperiod) <= 1,
                np.pi / 4 * np.cos(np.pi / 2 * (distances / timeperiod)),
                0,
            )
        )
    else:
        raise ValueError("kernel argument must be int from 0 to 5")
    # weights = weights.astype(close.dtype)
    weights_sum = weights.sum()
    nwma = np.array(
        [
            (weights @ close[i - timeperiod + 1 : i + 1]) / weights_sum
            for i in range(timeperiod - 1, len(close))
        ]
    )
    return np.concatenate((np.zeros(timeperiod - 1), nwma))


@jit(nopython=True, nogil=True, cache=True)
def LWMA(close: np.ndarray, period: int) -> np.ndarray:
    close = np.ascontiguousarray(close)
    weights = np.ascontiguousarray(np.arange(1, period + 1, dtype=close.dtype))
    weights_sum = weights.sum()
    lwma = np.array(
        [
            np.dot(close[i - period + 1 : i + 1], weights) / weights_sum
            for i in range(period - 1, len(close))
        ]
    )
    return np.concatenate((np.zeros(period - 1), lwma))


# @feature_timeit
@jit(nopython=True, nogil=True, cache=True)
def MGD(close: np.ndarray, period: int) -> np.ndarray:
    md = np.empty_like(close)
    md[0] = close[0]
    for i in range(1, len(close)):
        if md[i - 1] != 0:
            denominator = md[i - 1]
        else:
            denominator = 1.0
        md[i] = md[i - 1] + (close[i] - md[i - 1]) / (
            period * np.power((close[i] / denominator), 4)
        )
    return md


### It behaves differently depending on close len
# @feature_timeit
@jit(nopython=True, nogil=True, cache=True)
def VIDYA(close: np.ndarray, k: np.ndarray, period: int) -> np.ndarray:
    alpha = 2 / (period + 1)
    # k = talib.CMO(close, period)
    k = np.abs(k) / 100
    VIDYA = np.empty_like(close)
    VIDYA[period - 1] = close[period - 1]
    for i in range(period, len(close)):
        VIDYA[i] = alpha * k[i] * close[i] + (1 - alpha * k[i]) * VIDYA[i - 1]
    return VIDYA


# @feature_timeit
def FBA(close: np.ndarray, period: int) -> np.ndarray:
    fibs = []
    a, b = 1, 2
    while b <= period:
        fibs.append(b)
        a, b = b, a + b
    moving_averages = np.array([talib.EMA(close, i) for i in fibs]) / 100
    return (np.sum(moving_averages, axis=0) / len(fibs)) * 100


# @jit(nopython=True, nogil=True, cache=True)
def CWMA(close, weights, period):
    """Custom Weighted Moving Average"""
    cwma = np.zeros_like(close)
    window_weight_sum = np.sum(weights)
    window_prod_sum = np.sum(close[:period] * weights)
    cwma[period - 1] = window_prod_sum / window_weight_sum
    for i in range(period, len(close)):
        # print(f'window_weight_sum {window_weight_sum}')
        # print(f'window_prod_sum {window_prod_sum}')
        window_prod_sum = np.sum(close[i - period : i] * weights)
        cwma[i] = window_prod_sum / window_weight_sum
        # print(f'cwma {cwma[i]}')
    return cwma


# @jit(nopython=True, nogil=True, cache=True)
def FWMA(close, period):
    """Fibonacci Weighted Moving Average"""
    print(f"fibs {fib_to(period + 1, normalization=True)[1:]}")
    return CWMA(close, fib_to(period + 1, normalization=True)[1:], period)
