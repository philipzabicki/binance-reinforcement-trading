import glob
import os
from os import path

import pandas as pd
from talib import AD, TRANGE

from definitions import MODELING_DATASET_DIR
from utils.feature_generation import (
    custom_StochasticOscillator,
    custom_ChaikinOscillator,
    custom_MACD,
    custom_keltner_channel_signal,
)
from utils.get_data import by_BinanceVision
from utils.ta_tools import (
    StochasticOscillator_threshold_cross_signal,
    StochasticOscillator_threshold_signal,
    StochasticOscillator_mid_cross_signal,
    MACD_cross_signal,
    MACD_zero_cross_signal,
    MACD_histogram_reversal_signal,
    ChaikinOscillator_signal,
)

TICKER = "BTCUSDT"
ITV = "1h"
MARKET_TYPE = "spot"
DATA_TYPE = "klines"


def process_indicator_file(
        file_path, indicator_func, params_mapping, indicator_name_prefix
):
    params_df = pd.read_csv(file_path)
    for idx, row in params_df.iterrows():
        osc_params = [row[col] for col in params_mapping]

        # Proces dla Stochastic Oscillator – wybieramy odpowiednią funkcję sygnałową
        if indicator_func.__name__ == "custom_StochasticOscillator":
            slowK, slowD = indicator_func(ohlcv, *osc_params)
            if "mid" in indicator_name_prefix:
                # Oczekujemy kolumny 'mid_level' w CSV
                signals = StochasticOscillator_mid_cross_signal(
                    slowK, slowD, mid_level=row["mid_level"]
                )
            elif "threshold" in indicator_name_prefix:
                signals = StochasticOscillator_threshold_signal(
                    slowK,
                    slowD,
                    oversold_threshold=row["oversold_threshold"],
                    overbought_threshold=row["overbought_threshold"],
                )
            else:
                # Domyślnie traktujemy jako "cross" (czyli threshold_cross)
                signals = StochasticOscillator_threshold_cross_signal(
                    slowK,
                    slowD,
                    oversold_threshold=row["oversold_threshold"],
                    overbought_threshold=row["overbought_threshold"],
                )
            df[f"{indicator_name_prefix}_signal_{idx}"] = signals

        # Proces dla Chaikin Oscillator
        elif indicator_func.__name__ == "custom_ChaikinOscillator":
            osc_value = indicator_func(adl, *osc_params)
            signals = ChaikinOscillator_signal(osc_value)
            df[f"{indicator_name_prefix}_signal_{idx}"] = signals

        # Proces dla MACD – wybieramy funkcję sygnałową wg nazwy pliku
        elif indicator_func.__name__ == "custom_MACD":
            macd, macd_signal = indicator_func(ohlcv, *osc_params)
            if "zero" in indicator_name_prefix:
                signals = MACD_zero_cross_signal(macd, macd_signal)
            elif "histogram" in indicator_name_prefix:
                signals = MACD_histogram_reversal_signal(macd - macd_signal)
            else:  # domyślnie "cross"
                signals = MACD_cross_signal(macd, macd_signal)
            df[f"{indicator_name_prefix}_signal_{idx}"] = signals

        # Proces dla Keltner – sygnały są już w "surowej" postaci
        elif indicator_func.__name__ == "custom_keltner_channel_signal":
            signals = indicator_func(ohlcv, trange, *osc_params)
            df[f"{indicator_name_prefix}_signal_{idx}"] = signals

        else:
            # Domyślnie – funkcja zwraca już sygnały
            signals = indicator_func(ohlcv, *osc_params)
            df[f"{indicator_name_prefix}_signal_{idx}"] = signals


if __name__ == "__main__":
    # Wczytanie danych OHLCV
    df = by_BinanceVision(
        ticker=TICKER,
        interval=ITV,
        market_type=MARKET_TYPE,
        data_type=DATA_TYPE,
        start_date="2018-01-01 00:00:00",
        end_date="2025-01-01 00:00:00",
        split=False,
        delay=1_000_000,
    )
    # Zakładamy, że DataFrame zawiera kolumny: open, high, low, close, volume
    ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].to_numpy()

    # Obliczenie ADL i TRANGE – potrzebne do Chaikina oraz Keltnera
    adl = AD(
        df["High"].values, df["Low"].values, df["Close"].values, df["Volume"].values
    )
    trange = TRANGE(*ohlcv[:, 1:4].T.astype(float))

    base_folder = r"reports\feature_fits"

    # Przetwarzanie pliku Chaikin Oscillator
    chaikin_file = os.path.join(base_folder, "chaikin_oscillator.csv")
    process_indicator_file(
        file_path=chaikin_file,
        indicator_func=custom_ChaikinOscillator,
        params_mapping=["fast_period", "slow_period", "fast_ma_type", "slow_ma_type"],
        indicator_name_prefix="chaikin",
    )

    # Przetwarzanie plików MACD – uwzględniamy wszystkie typy sygnałów
    macd_files = [
        "macd_cross.csv",
        "macd_zero_cross.csv",
        "macd_histogram_reversal.csv",
    ]
    for macd_file_name in macd_files:
        macd_file = os.path.join(base_folder, macd_file_name)
        process_indicator_file(
            file_path=macd_file,
            indicator_func=custom_MACD,
            params_mapping=[
                "fast_source",
                "slow_source",
                "fast_period",
                "slow_period",
                "signal_period",
                "fast_ma_type",
                "slow_ma_type",
                "signal_ma_type",
            ],
            indicator_name_prefix=os.path.splitext(macd_file_name)[0],
        )

    # Przetwarzanie plików Stochastic Oscillator – również mamy kilka wariantów
    stoch_files = [
        "stoch_osc_cross.csv",
        "stoch_osc_mid_cross.csv",
        "stoch_osc_threshold.csv",
    ]
    for stoch_file_name in stoch_files:
        stoch_file = os.path.join(base_folder, stoch_file_name)
        process_indicator_file(
            file_path=stoch_file,
            indicator_func=custom_StochasticOscillator,
            params_mapping=[
                "fastK_period",
                "slowK_period",
                "slowD_period",
                "slowK_ma_type",
                "slowD_ma_type",
            ],
            indicator_name_prefix=os.path.splitext(stoch_file_name)[0],
        )

    # Przetwarzanie plików dla Keltner – pliki znajdują się w podfolderze "ma_band_action_fits"
    ma_folder = os.path.join(base_folder, "ma_band_action_fits")
    ma_files = glob.glob(os.path.join(ma_folder, "*.csv"))
    for file in ma_files:
        process_indicator_file(
            file_path=file,
            indicator_func=custom_keltner_channel_signal,
            params_mapping=[
                "ma_type",
                "ma_period",
                "atr_ma_type",
                "atr_period",
                "atr_multi",
                "source",
            ],
            indicator_name_prefix=os.path.splitext(os.path.basename(file))[0],
        )

    print(df.head())
    df.to_csv(path.join(MODELING_DATASET_DIR, "modeling.csv"), index=False)
