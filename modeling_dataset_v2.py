import ast
from datetime import time
from os import path

import re
import numpy as np
import pandas as pd
from talib import AD, TRANGE

from definitions import MODELING_DATASET_DIR
from utils.feature_functions import add_scaled_candle_patterns_indicators
from utils.feature_generation import (
    custom_StochasticOscillator,
    custom_ChaikinOscillator,
    custom_MACD,
    custom_keltner_channel_signal,
    add_volume_profile_fixed_range,
    compute_session_feature,
    get_triple_witching_timestamps,
    triple_witching_tri_feature,
    add_symbolic_reg_preds
)
from utils.get_data import by_BinanceVision
from utils.ta_tools import (
    ChaikinOscillator_signal,
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
    MACD_lines_cross_with_zero,
    MACD_lines_cross,
    MACD_lines_approaching_cross_with_zero,
    MACD_lines_approaching_cross,
    MACD_signal_line_zero_cross,
    MACD_line_zero_cross,
    MACD_histogram_reversal,
)

TICKER = "BTCUSDT"
ITV = "1h"
MARKET_TYPE = "spot"
DATA_TYPE = "klines"

TRAIN_START = pd.Timestamp('2018-01-01', tz='UTC')
TRAIN_END = pd.Timestamp('2023-12-31', tz='UTC')
TEST_START = pd.Timestamp('2024-01-01', tz='UTC')

PARAM_DIR = r"reports\feature_fits_quick"
# Optimal parameters files
CHAIKIN_PARAMS_FILE = 'chaikin_osc_pop8192_iters15_modemix_h01.csv'
KELTNER_PARAMS_FILE = 'keltner_channel_pop8192_iters15_modemix_h01.csv'
MACD_PARAMS_FILE = 'macd_pop8192_iters15_modemix_h01.csv'
STOCH_PARAMS_FILE = 'stoch_osc_pop8192_iters15_modemix_h01.csv'

_NP_WRAP = re.compile(r"np\.\w+\((.*?)\)")

def literal_eval_np(x):
    """Zdejmuje np.int64(...), np.float64(...), np.str_(...) itd. i odpala ast.literal_eval."""
    if not isinstance(x, str):
        return x
    s = x
    # usuwamy wszystkie wrappery np.*(...) aż nic nie zostanie
    while 'np.' in s:
        s_new, n = _NP_WRAP.subn(r"\1", s)
        if n == 0:
            break
        s = s_new
    return ast.literal_eval(s)

sessions = {
    # Ameryka Północna
    'NYSE': {'start': time(9, 30), 'end': time(16, 0), 'timezone': 'America/New_York'},
    'NASDAQ': {'start': time(9, 30), 'end': time(16, 0), 'timezone': 'America/New_York'},
    # Europa
    'LSE': {'start': time(8, 0), 'end': time(16, 30), 'timezone': 'Europe/London'},
    'Xetra': {'start': time(9, 0), 'end': time(17, 30), 'timezone': 'Europe/Berlin'},
    # Azja i Pacyfik
    'TSE_Morning': {'start': time(9, 0), 'end': time(11, 30), 'timezone': 'Asia/Tokyo'},
    'TSE_Afternoon': {'start': time(12, 30), 'end': time(15, 30), 'timezone': 'Asia/Tokyo'},
    'SSE_Morning': {'start': time(9, 30), 'end': time(11, 30), 'timezone': 'Asia/Shanghai'},
    'SSE_Afternoon': {'start': time(13, 0), 'end': time(15, 0), 'timezone': 'Asia/Shanghai'},
    'BSE': {'start': time(9, 15), 'end': time(15, 30), 'timezone': 'Asia/Kolkata'},
    'ASX': {'start': time(10, 0), 'end': time(16, 0), 'timezone': 'Australia/Sydney'},
    'HOSE_Morning': {'start': time(9, 15), 'end': time(11, 30), 'timezone': 'Asia/Ho_Chi_Minh'},
    'HOSE_Afternoon': {'start': time(13, 0), 'end': time(14, 30), 'timezone': 'Asia/Ho_Chi_Minh'},
    'PSE_Morning': {'start': time(9, 30), 'end': time(12, 0), 'timezone': 'Asia/Manila'},
    'PSE_Afternoon': {'start': time(13, 0), 'end': time(14, 45), 'timezone': 'Asia/Manila'},
    'PSX': {'start': time(9, 32), 'end': time(15, 30), 'timezone': 'Asia/Karachi'},
    'SET_Morning': {'start': time(10, 0), 'end': time(12, 30), 'timezone': 'Asia/Bangkok'},
    'SET_Afternoon': {'start': time(14, 0), 'end': time(16, 30), 'timezone': 'Asia/Bangkok'},
    'IDX': {'start': time(9, 0), 'end': time(15, 50), 'timezone': 'Asia/Jakarta'},
    # Bliski Wschód i Afryka
    'DFM': {'start': time(10, 0), 'end': time(15, 0), 'timezone': 'Asia/Dubai'},
    'NSE_Nigeria': {'start': time(10, 0), 'end': time(14, 20), 'timezone': 'Africa/Lagos'},
    'BIST_Morning': {'start': time(9, 30), 'end': time(12, 30), 'timezone': 'Europe/Istanbul'},
    'BIST_Afternoon': {'start': time(14, 0), 'end': time(17, 30), 'timezone': 'Europe/Istanbul'},
    'NSE_Kenya': {'start': time(9, 0), 'end': time(15, 0), 'timezone': 'Africa/Nairobi'},
    # Ameryka Południowa
    'B3': {'start': time(10, 0), 'end': time(16, 55), 'timezone': 'America/Sao_Paulo'},
    'BCBA': {'start': time(11, 0), 'end': time(17, 0), 'timezone': 'America/Argentina/Buenos_Aires'},
    'BVC': {'start': time(9, 30), 'end': time(15, 55), 'timezone': 'America/Bogota'},
}

if __name__ == "__main__":
    df = by_BinanceVision(
        ticker=TICKER,
        interval=ITV,
        market_type=MARKET_TYPE,
        data_type=DATA_TYPE,
        split=False,
        delay=1_000_000,
    )
    print('Loaded base dataset:')
    print(df.describe())
    ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].to_numpy()

    ADL = AD(*ohlcv[:, 1:5].T)
    # trange = TRANGE(*ohlcv[:, 1:4].T.astype(float))

    ### Chaikin Oscillator
    chaikin_results = pd.read_csv(path.join(PARAM_DIR, CHAIKIN_PARAMS_FILE))
    chaikin_results['params'] = chaikin_results['params'].apply(literal_eval_np)
    # chaikin_results['params'] = chaikin_results['params'].apply(ast.literal_eval)

    for idx, row in chaikin_results.iterrows():
        print(f'Chaikin Oscillator idx {idx} matched {row["matched"]}')
        params = row['params']

        chaikin_oscillator = custom_ChaikinOscillator(
            ADL,
            fast_ma_type=params['fast_ma_type'],
            fast_period=params['fast_period'],
            slow_ma_type=params["slow_ma_type"],
            slow_period=params['slow_period'],
        )
        signals = np.array(ChaikinOscillator_signal(chaikin_oscillator)).astype(int)
        col_name = (
            f"Chaikin_"
            f"fma{params['fast_ma_type']}_fp{params['fast_period']}_"
            f"sma{params['slow_ma_type']}_sp{params['slow_period']}"
        )
        df[col_name] = signals

    ### Keltner channel
    keltner_results = pd.read_csv(path.join(PARAM_DIR, KELTNER_PARAMS_FILE))
    keltner_results['params'] = keltner_results['params'].apply(literal_eval_np)
    # keltner_results['params'] = keltner_results['params'].apply(ast.literal_eval)

    for idx, row in keltner_results.iterrows():
        print(f'Keltner channel idx {idx} matched {row["matched"]}')
        params = row['params']

        signals = custom_keltner_channel_signal(
            ohlcv=ohlcv,
            true_range=TRANGE(*df.to_numpy()[:, 2:5].T.astype(float)),
            ma_type=params["ma_type"],
            ma_period=params["ma_period"],
            atr_ma_type=params["atr_ma_type"],
            atr_period=params["atr_period"],
            atr_multi=params["atr_multi"],
            source=params["source"],
        )

        signals = np.where(signals >= 1, 1, np.where(signals <= -1, -1, 0)).astype(int)
        col_name = (
            f"Keltner_"
            f"ma{params['ma_type']}_p{params['ma_period']}_"
            f"atrma{params['atr_ma_type']}_atrp{params['atr_period']}_atrmul{params['atr_multi']}"
            f"{params['source']}"
        )
        df[col_name] = signals

    ### MACD
    macd_results = pd.read_csv(path.join(PARAM_DIR, MACD_PARAMS_FILE))
    macd_results['params'] = macd_results['params'].apply(literal_eval_np)
    # macd_results['params'] = macd_results['params'].apply(ast.literal_eval)

    for idx, row in macd_results.iterrows():
        print(f'MACD idx {idx} matched {row["matched"]}')
        params = row['params']

        macd, macd_signal = custom_MACD(
            ohlcv,
            fast_source=params["fast_source"],
            slow_source=params["slow_source"],
            fast_ma_type=params["fast_ma_type"],
            fast_period=params["fast_period"],
            slow_ma_type=params["slow_ma_type"],
            slow_period=params["slow_period"],
            signal_ma_type=params["signal_ma_type"],
            signal_period=params["signal_period"],
        )

        signal_func_mapping = {
            "lines_cross_with_zero": MACD_lines_cross_with_zero,
            "lines_cross": MACD_lines_cross,
            "lines_approaching_cross_with_zero": MACD_lines_approaching_cross_with_zero,
            "lines_approaching_cross": MACD_lines_approaching_cross,
            "signal_line_zero_cross": MACD_signal_line_zero_cross,
            "MACD_line_zero_cross": MACD_line_zero_cross,
            "histogram_reversal": MACD_histogram_reversal,
        }
        func = signal_func_mapping[params["signal_type"]]
        signals = np.array(func(macd, macd_signal)).astype(int)
        col_name = (
            f"MACD_"
            f"{params['signal_type']}_fs{params['fast_source']}_ss{params['slow_source']}_"
            f"fmat{params['fast_ma_type']}_fp{params['fast_period']}_"
            f"smat{params['slow_ma_type']}_sp{params['slow_period']}_"
            f"sigmat{params['signal_ma_type']}_sigp{params['signal_period']}"
        )
        df[col_name] = signals

    ### Stochastic Oscillator
    stoch_results = pd.read_csv(path.join(PARAM_DIR, STOCH_PARAMS_FILE))
    stoch_results['params'] = stoch_results['params'].apply(literal_eval_np)
    # stoch_results['params'] = stoch_results['params'].apply(ast.literal_eval)

    for idx, row in stoch_results.iterrows():
        print(f'Stochastic Oscillator idx {idx} matched {row["matched"]}')
        params = row['params']

        slowK, slowD = custom_StochasticOscillator(
            ohlcv,
            fastK_period=params["fastK_period"],
            slowK_period=params["slowK_period"],
            slowD_period=params["slowD_period"],
            slowK_ma_type=params["slowK_ma_type"],
            slowD_ma_type=params["slowD_ma_type"],
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

        func = signal_func_mapping[params["signal_type"]]
        signals = np.array(
            func(
                k_line=slowK,
                d_line=slowD,
                oversold_threshold=params["oversold_threshold"],
                overbought_threshold=params["overbought_threshold"],
            )
        ).astype(int)
        df[col_name] = signals

    df = add_volume_profile_fixed_range(
        df,
        price_min=1,
        price_max=1_000_000,
        step=1,
        bins_back=100,
        bins_forward=100,
        vwap_col='HL2'
    )

    df = add_symbolic_reg_preds(df)
    df = add_scaled_candle_patterns_indicators(df)

    df['Opened'] = pd.to_datetime(df['Opened'], errors='coerce', utc=True)
    for session, info in sessions.items():
        df[f'session_{session}_feature'] = df.apply(compute_session_feature, session_info=info, axis=1)

    tw_hours = get_triple_witching_timestamps(df["Opened"].min(),
                                              df["Opened"].max())

    df["triple_witching_tri"] = df["Opened"].apply(
        lambda ts: triple_witching_tri_feature(ts, tw_hours)
    )
    df["is_triple_witching_hour"] = df["Opened"].isin(tw_hours).astype(int)

    df.to_csv(path.join(MODELING_DATASET_DIR, "full_modeling.csv"), index=False)

    df_train = df[(df['Opened'] >= TRAIN_START) & (df['Opened'] <= TRAIN_END)]
    df_train.to_csv(path.join(MODELING_DATASET_DIR, "train_modeling.csv"), index=False)

    df_test = df[df['Opened'] >= TEST_START]
    df_test.to_csv(path.join(MODELING_DATASET_DIR, "test_modeling.csv"), index=False)
