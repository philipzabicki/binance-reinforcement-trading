import math
from datetime import timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import pytz
import talib
from pandas.tseries.offsets import Week

from .ta_tools import get_ma_from_source, get_1D_MA, any_ma_sig

OHLCV_BYTES = None
OHLCV_SHAPE = None
TRANGE_BYTES = None
TRANGE_SHAPE = None
ADL_BYTES = None
ADL_SHAPE = None
UP_MOVE_BYTES = None
UP_MOVE_SHAPE = None
DOWN_MOVE_BYTES = None
DOWN_MOVE_SHAPE = None


def keltner_channel_initializer(ohlcv: np.ndarray):
    # print(f"keltner_channel_initializer ohlcv: {ohlcv}")
    ohlcv_initializer(ohlcv)
    global TRANGE_BYTES, TRANGE_SHAPE
    # Precompute the bytes representation and shape of the constant array.
    true_range = talib.TRANGE(*ohlcv[:, 1:4].T)
    TRANGE_BYTES = true_range.tobytes()
    TRANGE_SHAPE = true_range.shape


def adx_initializer(ohlcv: np.ndarray):
    # print(f"adx_initializer ohlcv: {ohlcv}")
    global TRANGE_BYTES, TRANGE_SHAPE, UP_MOVE_BYTES, UP_MOVE_SHAPE, DOWN_MOVE_BYTES, DOWN_MOVE_SHAPE
    # Precompute the byte representation and shape of the constant array.
    true_range = talib.TRANGE(*ohlcv[:, 1:4].T)
    TRANGE_BYTES = true_range.tobytes()
    TRANGE_SHAPE = true_range.shape

    prev_high = np.roll(ohlcv[:, 1], 1)
    prev_high[0] = ohlcv[0, 1]
    prev_low = np.roll(ohlcv[:, 2], 1)
    prev_low[0] = ohlcv[0, 2]

    high_diff = ohlcv[:, 1] - prev_high
    low_diff = prev_low - ohlcv[:, 2]
    up_move = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    down_move = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    UP_MOVE_BYTES = up_move.tobytes()
    UP_MOVE_SHAPE = up_move.shape
    DOWN_MOVE_BYTES = down_move.tobytes()
    DOWN_MOVE_SHAPE = down_move.shape



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


def custom_ADX(ohlcv,
               atr_period,
               posDM_period,
               negDM_period,
               adx_period,
               ma_type_atr,
               ma_type_posDM,
               ma_type_negDM,
               ma_type_adx):
    global UP_MOVE_BYTES, UP_MOVE_SHAPE, DOWN_MOVE_BYTES, DOWN_MOVE_SHAPE
    if UP_MOVE_BYTES is None or UP_MOVE_SHAPE is None or DOWN_MOVE_BYTES is None or DOWN_MOVE_SHAPE is None:
        adx_initializer(ohlcv)
        # raise ValueError("Global OHLCV data not initialized!")

    TR_smooth = custom_ATR_cache(ma_type_atr, atr_period)
    pos_DM_smooth = get_1D_MA(np.frombuffer(UP_MOVE_BYTES, dtype=np.float64).reshape(UP_MOVE_SHAPE),
                              ma_type_posDM, posDM_period)
    neg_DM_smooth = get_1D_MA(np.frombuffer(DOWN_MOVE_BYTES, dtype=np.float64).reshape(DOWN_MOVE_SHAPE),
                              ma_type_negDM, negDM_period)

    # Ensure non-negative DM and safe denominators
    pos_DM_smooth = np.maximum(pos_DM_smooth, 0.0)
    neg_DM_smooth = np.maximum(neg_DM_smooth, 0.0)
    tiny = 1e-12  # safer than float64 eps for price scales

    # plus/minus DI as percents in a numerically safe way
    plus = np.divide(pos_DM_smooth, TR_smooth, out=np.zeros_like(TR_smooth), where=np.abs(TR_smooth) > tiny)
    minus = np.divide(neg_DM_smooth, TR_smooth, out=np.zeros_like(TR_smooth), where=np.abs(TR_smooth) > tiny)
    plus_DI = 100.0 * plus
    minus_DI = 100.0 * minus

    # Robust DX in [0, 1], then scale by 100
    denom = np.abs(plus_DI) + np.abs(minus_DI)
    DX = np.divide(np.abs(plus_DI - minus_DI), denom, out=np.zeros_like(denom), where=denom > tiny)
    DX = np.clip(DX, 0.0, 1.0)
    DX_raw = 100.0 * DX

    return get_1D_MA(DX_raw, ma_type_adx, adx_period), plus_DI, minus_DI


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
        price_min=1,
        price_max=1_000_000,
        step=5,
        bins_back=10,
        bins_forward=10,
        vwap_col='VWAP'
):
    if vwap_col == 'HL2':
        df['HL2'] = (df['High'] + df['Low']) / 2

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
        end = get_bin_idx(high_i)
        volume_profile[start:end + 1] += vol_i

    if vwap_col == 'HL2':
        df.drop(columns='HL2', inplace=True)

    return df


def compute_session_feature(row, session_info, time_col='Opened'):
    timestamp = row[time_col]
    if pd.isnull(timestamp):
        return np.nan

    # Ensure timestamp is tz-aware (UTC as base)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    tz = pytz.timezone(session_info['timezone'])
    local_time = timestamp.astimezone(tz)
    s, e = session_info['start'], session_info['end']
    session_start = local_time.replace(hour=s.hour, minute=s.minute, second=0, microsecond=0)
    session_end = local_time.replace(hour=e.hour, minute=e.minute, second=0, microsecond=0)

    # Przed sesją
    if local_time < session_start:
        prev_day = local_time - timedelta(days=1)
        prev_close = session_end.replace(year=prev_day.year, month=prev_day.month, day=prev_day.day)
        off_time = (session_start - prev_close).total_seconds()
        elapsed = (local_time - prev_close).total_seconds()
        x = elapsed / off_time
        return -math.sin(math.pi * x)
    # Po sesji
    elif local_time > session_end:
        next_day = local_time + timedelta(days=1)
        next_open = session_start.replace(year=next_day.year, month=next_day.month, day=next_day.day)
        off_time = (next_open - session_end).total_seconds()
        elapsed = (local_time - session_end).total_seconds()
        x = elapsed / off_time
        return -math.sin(math.pi * x)
    # W trakcie sesji
    else:
        in_time = (session_end - session_start).total_seconds()
        elapsed = (local_time - session_start).total_seconds()
        x = elapsed / in_time
        return math.sin(math.pi * x)


def get_triple_witching_timestamps(start, end):
    if start.tzinfo is None:
        start = start.tz_localize('UTC')
    if end.tzinfo is None:
        end = end.tz_localize('UTC')

    tz = pytz.timezone('America/New_York')
    timestamps_utc = []
    # +2 lata to bufor, żeby na pewno załapać się na "następny"
    for year in range(start.year, end.year + 2):
        for month in (3, 6, 9, 12):
            d = pd.Timestamp(year=year, month=month, day=1, tz=tz)
            first_friday = d + Week(weekday=4)
            third_friday = first_friday + Week(2)
            local_dt = third_friday.replace(hour=15, minute=0, second=0)
            utc_dt = local_dt.tz_convert('UTC')
            timestamps_utc.append(utc_dt)

    # zatrzymujemy wszystko od start w górę…
    dates = sorted([d for d in timestamps_utc if d >= start])

    # …i JEŚLI ostatnia ≤ end, dorzucamy pierwszy punkt po end
    if dates and dates[-1] <= end:
        for d in timestamps_utc:
            if d > end:
                dates.append(d)
                break

    return dates


def triple_witching_tri_feature(ts: pd.Timestamp,
                                tw_datetimes: list[pd.Timestamp]) -> float:
    """
    Jednostkowa, zamknięta w [0, 1] funkcja:
      • 1 dokładnie w chwili 3 Wiedźm,
      • 0 w połowie odległości pomiędzy kolejnymi 3 Wiedźmami,
      • liniowo ↓ i potem ↑ (trójkąt).
    """
    if pd.isnull(ts):
        return np.nan
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    prev_tw = max((d for d in tw_datetimes if d <= ts), default=None)
    next_tw = min((d for d in tw_datetimes if d > ts), default=None)

    if prev_tw is None or next_tw is None:
        return np.nan

    total_sec = (next_tw - prev_tw).total_seconds()
    elapsed = (ts - prev_tw).total_seconds()
    x = elapsed / total_sec  # 0 → prev TW, 1 → next TW

    return 2.0 * abs(x - 0.5)  # 1 na końcach, 0 w środku


def is_exact_triple_witching_hour(ts, witching_datetimes):
    # Dokładne dopasowanie timestampu do triple witching hour
    return int(ts in witching_datetimes)


def add_symbolic_reg_preds(df):
    temp_cols = []
    df["body"] = (df["Close"] - df["Open"]).abs()
    df["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
    temp_cols.extend(["body", "upper_shadow", "lower_shadow"])
    for col in ["Open", "High", "Low", "Close", "Volume", "body", "upper_shadow", "lower_shadow"]:
        for l in range(1, 4):
            col_name = f"{col}_lag{l}"
            df[col_name] = df[col].shift(l)
            temp_cols.append(col_name)
    # boost 0
    df["y_prev1_neg"] = 0.49966812 / (
            np.less_equal(
                df["Close"],
                np.less_equal(df["upper_shadow"] - 0.99753463, 5.7205725)
                + df["Open"] * 0.99753463,
            )
            - 0.49966812
    )
    df["y_prev1_pos"] = np.negative(
        (
                np.greater_equal(
                    df["Open"],
                    df["Close"]
                    - np.maximum(
                        df["upper_shadow"] * 2 + np.abs(0.7457582 * df["upper_shadow"]),
                        np.abs(df["Open"] / df["Volume"]),
                    ),
                )
                * 2.001982
        )
        - 1.0
    )

    # boost 1
    df["y_prev2_neg"] = np.maximum(
        df["y_prev1_neg"],
        np.minimum(
            np.minimum(
                np.minimum(df["Open_lag1"] - df["Close_lag1"], df["lower_shadow_lag1"])
                - (
                        np.greater_equal(0.87595224, -2.0697224)
                        + np.abs(np.maximum(df["Close"] - df["Low_lag1"], 0))
                ),
                np.greater_equal(
                    df["upper_shadow_lag1"] + df["upper_shadow"] * df["body_lag1"],
                    df["lower_shadow"],
                ),
            ),
            -df["y_prev1_pos"],
        ),
    )
    df["y_prev2_pos"] = np.minimum(
        np.maximum(
            -1.0,
            df["y_prev1_pos"] * df["body_lag1"]
            + df["lower_shadow"]
            * (
                    (0.14052474 * df["lower_shadow"] + df["Close"])
                    - ((df["body_lag1"] + df["Open_lag1"]) / 0.99850434)
                    - np.greater_equal(
                df["upper_shadow"],
                np.maximum(df["y_prev1_pos"], 0) + df["body_lag1"],
            )
            ),
        ),
        -df["y_prev1_neg"],
    )

    # boost 2
    df["y_prev3_neg"] = np.minimum(
        np.maximum(
            1.6965557
            + (
                    np.less_equal(
                        np.minimum(
                            df["High_lag2"]
                            - np.minimum(
                                df["Open_lag2"], df["High"] - df["upper_shadow_lag1"]
                            )
                            - df["lower_shadow"],
                            df["lower_shadow_lag3"],
                        ),
                        df["Close_lag2"],
                    )
                    - np.maximum(df["Low_lag1"], 0)
            ),
            df["y_prev2_pos"],
        ),
        df["y_prev2_neg"],
    )
    df["y_prev3_pos"] = np.minimum(
        np.minimum(
            df["y_prev2_pos"],
            np.maximum(
                df["Close"]
                - np.negative(
                    (df["upper_shadow_lag2"] * -0.56393766) - df["Close_lag3"]
                ),
                0,
            )
            - np.less_equal(
                df["Volume_lag3"], df["Open_lag3"] * np.negative(-0.56393766)
            ),
        ),
        (df["lower_shadow_lag1"] + df["lower_shadow_lag3"]) - df["y_prev2_pos"],
    )

    df.drop(columns=temp_cols, inplace=True)
    return df
