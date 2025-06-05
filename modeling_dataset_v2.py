from os import path

import pandas as pd
import numpy as np
import ast
from talib import AD, TRANGE

from definitions import MODELING_DATASET_DIR
from utils.feature_generation import (
    custom_StochasticOscillator,
    custom_ChaikinOscillator,
    custom_MACD,
    custom_keltner_channel_signal,
    add_volume_profile_fixed_range
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
START_DATE = pd.Timestamp('2018-01-01')
END_DATE = pd.Timestamp('2024-12-31')
PARAM_DIR = r"reports\feature_fits_quick"
# Optimal parameters files
CHAIKIN_PARAMS_FILE = 'chaikin_osc_pop2048_iters25_modemix.csv'
KELTNER_PARAMS_FILE = 'keltner_channel_pop2048_iters25_modemix.csv'
MACD_PARAMS_FILE = 'macd_pop2048_iters25_modemix.csv'
STOCH_PARAMS_FILE = 'stoch_osc_pop1024_iters25_modemix.csv'


if __name__ == "__main__":
    df = by_BinanceVision(
        ticker=TICKER,
        interval=ITV,
        market_type=MARKET_TYPE,
        data_type=DATA_TYPE,
        # start_date="2018-01-01 00:00:00",
        # end_date="2025-01-01 00:00:00",
        split=False,
        delay=1_000_000,
    )
    print(df.describe())
    ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].to_numpy()

    ADL = AD(*ohlcv[:, 1:5].T)
    # trange = TRANGE(*ohlcv[:, 1:4].T.astype(float))

    ### Chaikin Oscillator
    chaikin_results = pd.read_csv(path.join(PARAM_DIR, CHAIKIN_PARAMS_FILE))
    chaikin_results['params'] = chaikin_results['params'].apply(ast.literal_eval)

    for idx, row in chaikin_results.iterrows():
        print(f'row {row}')
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
    keltner_results['params'] = keltner_results['params'].apply(ast.literal_eval)

    for idx, row in keltner_results.iterrows():
        print(f'row {row}')
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
    macd_results['params'] = macd_results['params'].apply(ast.literal_eval)

    for idx, row in macd_results.iterrows():
        print(f'row {row}')
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
        stoch_results['params'] = stoch_results['params'].apply(ast.literal_eval)

        for idx, row in stoch_results.iterrows():
            print(f'row {row}')
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
        step=5,
        bins_back=100,
        bins_forward=100,
        vwap_col='HL2'
    )

    df['Opened'] = pd.to_datetime(df['Opened'], errors='coerce')
    df = df[(df['Opened'] >= START_DATE) & (df['Opened'] <= END_DATE)]

    df.to_csv(path.join(MODELING_DATASET_DIR, "modeling_test.csv"), index=False)