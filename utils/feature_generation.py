import numpy as np
import talib

from .ta_tools import get_ma_from_source, get_1D_MA, any_ma_sig


def custom_StochasticOscillator(ohlcv, fastK_period, slowK_period, slowD_period, slowK_ma_type, slowD_ma_type):
    fastK, _ = talib.STOCHF(*ohlcv[:, 1:4].T.astype(float), fastk_period=fastK_period, fastd_period=1,
                            fastd_matype=0)
    slowK = fastK if slowK_period == 1 else get_1D_MA(fastK, slowK_ma_type, slowK_period)
    slowD = get_1D_MA(slowK, slowD_ma_type, slowD_period)
    return slowK, slowD


def custom_ChaikinOscillator(adl, fast_period, slow_period, fast_ma_type, slow_ma_type):
    return get_1D_MA(adl, fast_ma_type, fast_period) - get_1D_MA(adl, slow_ma_type, slow_period)


def custom_MACD(ohlcv, fast_source, slow_source, fast_period, slow_period, signal_period,
                fast_ma_type, slow_ma_type, signal_ma_type):
    macd = get_ma_from_source(ohlcv, fast_ma_type, fast_period, fast_source) - get_ma_from_source(ohlcv, slow_ma_type,
                                                                                                  slow_period,
                                                                                                  slow_source)
    return macd, get_1D_MA(macd, signal_ma_type, signal_period)


def custom_keltner_channel_signal(np_df: np.ndarray,
                                  true_range: np.ndarray,
                                  ma_type: int,
                                  ma_period: int,
                                  atr_ma_type: int,
                                  atr_period: int,
                                  atr_multi: float,
                                  source: str):
    return any_ma_sig(np_close=np_df[:, 3],
                      np_xMA=get_ma_from_source(np_df, ma_type, ma_period, source),
                      np_ATR=get_1D_MA(true_range, atr_ma_type, atr_period),
                      atr_multi=atr_multi)


def custom_ADX_classic_diff_ma(
        ohlcv: np.ndarray,
        adx_period: int,
        ma_type_tr: int,
        ma_type_posDM: int,
        ma_type_negDM: int,
        ma_type_dx: int
):
    high = ohlcv[:, 1]
    low = ohlcv[:, 2]
    close = ohlcv[:, 3]

    # 1) Surowy TR z TA-Lib (bez smoothingu)
    TR_raw = talib.TRANGE(high, low, close)

    # 2) Surowe +DM, -DM (bez smoothingu z TA-Lib, bo tam jest Wilder)
    prev_high = np.roll(high, 1)
    prev_high[0] = high[0]

    prev_low = np.roll(low, 1)
    prev_low[0] = low[0]

    up_move = high - prev_high
    down_move = prev_low - low

    pos_DM_raw = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0.0
    )
    neg_DM_raw = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0.0
    )

    # 3) Pierwsze wygładzenie TR, +DM, -DM -> get_1D_MA
    TR_smooth = get_1D_MA(TR_raw, ma_type_tr, adx_period)
    pos_DM_smooth = get_1D_MA(pos_DM_raw, ma_type_posDM, adx_period)
    neg_DM_smooth = get_1D_MA(neg_DM_raw, ma_type_negDM, adx_period)

    # 4) +DI, -DI (już po wygładzeniu TR, +DM, -DM)
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_DI = 100.0 * (pos_DM_smooth / TR_smooth)
        minus_DI = 100.0 * (neg_DM_smooth / TR_smooth)

    # 5) Surowy DX = 100 * (|+DI - -DI| / (+DI + -DI))
    with np.errstate(divide='ignore', invalid='ignore'):
        DX_raw = 100.0 * np.abs(plus_DI - minus_DI) / (plus_DI + minus_DI)

    # 6) Drugie wygładzenie: DX -> ADX
    #    (zamiast Wilder Smoothing, używamy get_1D_MA z ma_type_dx)
    adx = get_1D_MA(DX_raw, ma_type_dx, adx_period)

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
