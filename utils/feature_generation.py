from functools import lru_cache

import numpy as np
import talib

from .ta_tools import get_ma_from_source, get_1D_MA, any_ma_sig

OHLCV_BYTES = None
OHLCV_SHAPE = None
TRANGE_BYTES = None
TRANGE_SHAPE = None
ADL_BYTES = None
ADL_SHAPE = None


def keltner_channel_initializer(ohlcv: np.ndarray):
    # print(f"keltner_channel_initializer ohlcv: {ohlcv}")
    ohlcv_initializer(ohlcv)
    global TRANGE_BYTES, TRANGE_SHAPE
    # Precompute the bytes representation and shape of the constant array.
    true_range = talib.TRANGE(*ohlcv[:, 1:4].T)
    TRANGE_BYTES = true_range.tobytes()
    TRANGE_SHAPE = true_range.shape


def adl_initializer(ohlcv: np.ndarray):
    global ADL_BYTES, ADL_SHAPE
    adl = talib.AD(*ohlcv[:, 1:5].T)
    ADL_BYTES = adl.tobytes()
    ADL_SHAPE = adl.shape


def ohlcv_initializer(ohlcv: np.ndarray):
    global OHLCV_BYTES, OHLCV_SHAPE
    OHLCV_BYTES = ohlcv.tobytes()
    OHLCV_SHAPE = ohlcv.shape


@lru_cache(maxsize=2048)
def get_ma_from_source_cache(ma_type: int, ma_period: int, source: str) -> np.ndarray:
    global OHLCV_BYTES, OHLCV_SHAPE
    if OHLCV_BYTES is None or OHLCV_SHAPE is None:
        raise ValueError("Global OHLCV data not initialized!")
    ohlcv_array = np.frombuffer(OHLCV_BYTES, dtype=np.float64).reshape(OHLCV_SHAPE)
    return get_ma_from_source(ohlcv_array, ma_type, ma_period, source)


@lru_cache(maxsize=4096)
def get_adl_ma_cache(ma_type: int, period: int) -> np.ndarray:
    global ADL_BYTES, ADL_SHAPE
    if ADL_BYTES is None or ADL_SHAPE is None:
        raise ValueError("Global OHLCV data not initialized!")
    adl = np.frombuffer(ADL_BYTES, dtype=np.float64).reshape(ADL_SHAPE)
    return get_1D_MA(adl, ma_type, period)


@lru_cache(maxsize=1024)
def custom_ATR_cache(atr_ma_type: int, atr_period: int) -> np.ndarray:
    global TRANGE_BYTES, TRANGE_SHAPE
    if TRANGE_BYTES is None or TRANGE_SHAPE is None:
        raise ValueError("Global True Range data not initialized!")
    true_range = np.frombuffer(TRANGE_BYTES, dtype=np.float64).reshape(TRANGE_SHAPE)
    return get_1D_MA(true_range, atr_ma_type, atr_period)


@lru_cache(maxsize=8192)
def sochf_cache(fastk_period: int) -> np.ndarray:
    global OHLCV_BYTES, OHLCV_SHAPE
    if OHLCV_BYTES is None or OHLCV_SHAPE is None:
        raise ValueError("Global OHLCV data not initialized!")
    ohlcv_array = np.frombuffer(OHLCV_BYTES, dtype=np.float64).reshape(OHLCV_SHAPE)
    fastK, _ = talib.STOCHF(
        *ohlcv_array[:, 1:4].T,
        fastk_period=fastk_period,
        fastd_period=1,
        fastd_matype=0,
    )
    return fastK


def custom_keltner_channel_signal_cached(
        ohlcv: np.ndarray,
        ma_type: int,
        ma_period: int,
        atr_ma_type: int,
        atr_period: int,
        atr_multi: float,
        source: str,
):
    return any_ma_sig(
        np_close=ohlcv[:, 3],
        np_xMA=get_ma_from_source_cache(ma_type, ma_period, source),
        np_ATR=custom_ATR_cache(atr_ma_type, atr_period),
        atr_multi=atr_multi,
    )


def custom_MACD_cached(
        fast_source,
        slow_source,
        fast_period,
        slow_period,
        signal_period,
        fast_ma_type,
        slow_ma_type,
        signal_ma_type,
):
    macd = get_ma_from_source_cache(
        fast_ma_type, fast_period, fast_source
    ) - get_ma_from_source_cache(slow_ma_type, slow_period, slow_source)
    return macd, get_1D_MA(macd, signal_ma_type, signal_period)


def custom_StochasticOscillator_cached(
        fastK_period, slowK_period, slowD_period, slowK_ma_type, slowD_ma_type
):
    fastK = sochf_cache(fastK_period)
    slowK = (
        np.nan_to_num(fastK)
        if slowK_period == 1
        else get_1D_MA(fastK, slowK_ma_type, slowK_period)
    )
    slowD = get_1D_MA(slowK, slowD_ma_type, slowD_period)
    return slowK, slowD


def custom_ChaikinOscillator_cached(
        fast_period, slow_period, fast_ma_type, slow_ma_type
):
    return get_adl_ma_cache(fast_ma_type, fast_period) - get_adl_ma_cache(
        slow_ma_type, slow_period
    )


def custom_StochasticOscillator(
        ohlcv, fastK_period, slowK_period, slowD_period, slowK_ma_type, slowD_ma_type
):
    fastK, _ = talib.STOCHF(
        *ohlcv[:, 1:4].T, fastk_period=fastK_period, fastd_period=1, fastd_matype=0
    )
    slowK = (
        np.nan_to_num(fastK)
        if slowK_period == 1
        else get_1D_MA(fastK, slowK_ma_type, slowK_period)
    )
    slowD = get_1D_MA(slowK, slowD_ma_type, slowD_period)
    return slowK, slowD


def custom_ChaikinOscillator(adl, fast_period, slow_period, fast_ma_type, slow_ma_type):
    return get_1D_MA(adl, fast_ma_type, fast_period) - get_1D_MA(
        adl, slow_ma_type, slow_period
    )


def custom_MACD(
        ohlcv,
        fast_source,
        slow_source,
        fast_period,
        slow_period,
        signal_period,
        fast_ma_type,
        slow_ma_type,
        signal_ma_type,
):
    macd = get_ma_from_source(
        ohlcv, fast_ma_type, fast_period, fast_source
    ) - get_ma_from_source(ohlcv, slow_ma_type, slow_period, slow_source)
    return macd, get_1D_MA(macd, signal_ma_type, signal_period)


def custom_keltner_channel_signal(
        ohlcv: np.ndarray,
        true_range: np.ndarray,
        ma_type: int,
        ma_period: int,
        atr_ma_type: int,
        atr_period: int,
        atr_multi: float,
        source: str,
):
    return any_ma_sig(
        np_close=ohlcv[:, 3],
        np_xMA=get_ma_from_source(ohlcv, ma_type, ma_period, source),
        np_ATR=get_1D_MA(true_range, atr_ma_type, atr_period),
        atr_multi=atr_multi,
    )


def custom_ADX(
        ohlcv: np.ndarray,
        true_range: np.ndarray,
        atr_period: int,
        posDM_period: int,
        negDM_period: int,
        adx_period: int,
        ma_type_atr: int,
        ma_type_posDM: int,
        ma_type_negDM: int,
        ma_type_adx: int,
):
    # print('')
    high = ohlcv[:, 1]
    low = ohlcv[:, 2]

    prev_high = np.roll(high, 1)
    prev_high[0] = high[0]

    prev_low = np.roll(low, 1)
    prev_low[0] = low[0]

    up_move = high - prev_high
    down_move = prev_low - low

    pos_DM_raw = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_DM_raw = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    TR_smooth = get_1D_MA(true_range, ma_type_atr, atr_period)
    # print(f"TR_smooth {np.quantile(TR_smooth, [0.25, 0.5, 0.75])}")
    pos_DM_smooth = get_1D_MA(pos_DM_raw, ma_type_posDM, posDM_period)
    # print(f"pos_DM_smooth {np.quantile(pos_DM_smooth, [0.25, 0.5, 0.75])}")
    neg_DM_smooth = get_1D_MA(neg_DM_raw, ma_type_negDM, negDM_period)
    # print(f"neg_DM_smooth {np.quantile(neg_DM_smooth, [0.25, 0.5, 0.75])}")

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_DI = 100.0 * np.nan_to_num(
            pos_DM_smooth / TR_smooth, posinf=0.0, neginf=0.0
        )
        minus_DI = 100.0 * np.nan_to_num(
            neg_DM_smooth / TR_smooth, posinf=0.0, neginf=0.0
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        DX_raw = 100.0 * np.nan_to_num(
            np.abs(plus_DI - minus_DI) / (plus_DI + minus_DI)
        )

    adx = get_1D_MA(DX_raw, ma_type_adx, adx_period)
    # print(f"plus_DI {np.quantile(plus_DI, [0.25, 0.5, 0.75])}")
    # print(f"minus_DI {np.quantile(minus_DI, [0.25, 0.5, 0.75])}")
    # print(f"DX_raw {np.quantile(DX_raw, [0.25, 0.5, 0.75])}")

    # for name, data in [('ADX', adx), ('+DI', plus_DI), ('-DI', minus_DI)]:
    #     q = np.quantile(data, [0.25, 0.5, 0.75])
    #     print(
    #         f"{name} Kwantyle 25%: {q[0]:.2f}, 50% (Mediana): {q[1]:.2f}, 75%: {q[2]:.2f} Min: {np.min(data):.2f}, Max: {np.max(data):.2f}")
    return adx, plus_DI, minus_DI

# def custom_keltner_channel_signal_old(np_df: np.ndarray, ma_type: int, ma_period: int, atr_period: int, atr_multi: float,
#                                   source: str):
#     def any_ma_sig(np_close: np.ndarray, np_xMA: np.ndarray, np_ATR: np.ndarray, atr_multi: float = 1.0) -> np.ndarray:
#         return ((np_xMA - np_close) / np_ATR) / atr_multi
#
#     atr = talib.ATR(np_df[:, 1], np_df[:, 2], np_df[:, 3], atr_period)
#     try:
#         return any_ma_sig(np_df[:, 3],
#                           get_ma_from_source(np_df, ma_type, ma_period, source),
#                           atr,
#                           atr_multi)
#     except TypeError:
#         print(f'len(np_df) {len(np_df)}')
#         print(f'np_df[:, 3] {np_df[:, 3]}')

def add_volume_profile_fixed_range(
        df,
        price_min = 1,
        price_max = 1_000_000,
        step = 5,
        bins_back = 10,
        bins_forward = 10,
        vwap_col='VWAP'
    ):
    if vwap_col == 'HL2':
        df['HL2'] = (df['High'] + df['Low'])/2

    price_levels = np.arange(price_min, price_max + step, step, dtype=float)
    bins = len(price_levels) - 1
    volume_profile = np.zeros(bins, dtype=float)

    df = df.copy().reset_index(drop=True)
    shifts = [s for s in range(-bins_back, bins_forward + 1) if s != 0]
    for s in shifts:
        df[f"VP_bin_pct_{s:+}"] = np.nan

    def get_bin_idx(price):
        idx = np.searchsorted(price_levels, price, side="right") - 1
        return np.clip(idx, 0, bins - 1)

    for i, (low_i, high_i, vwap_i, vol_i) in df[["Low", "High", vwap_col, "Volume"]].iterrows():
        bin_idx = get_bin_idx(vwap_i)
        curr_vol = volume_profile[bin_idx]

        for s in shifts:
            nb = np.clip(bin_idx + s, 0, bins - 1)
            neigh_vol = volume_profile[nb]
            col = f"VP_bin_pct_{s:+}"
            df.at[i, col] = 0.0 if curr_vol == 0 else (neigh_vol - curr_vol) / curr_vol

        start = get_bin_idx(low_i)
        end   = get_bin_idx(high_i)
        volume_profile[start:end + 1] += vol_i

    if vwap_col == 'HL2':
        df.drop(columns='HL2', inplace=True)

    return df