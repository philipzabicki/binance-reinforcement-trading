import glob
import os

import numpy as np
import pandas as pd

from environments.action_optimize_env import SignalCleanSpotEnv
from utils.get_data import by_BinanceVision

relative_path = os.path.join("..", "reports", "optimal_actions")
absolute_path = os.path.abspath(relative_path)
FILE_PATTERN = os.path.join(absolute_path, 'part_**_actions_rew*.csv')

TICKER = 'BTCUSDT'
ITV = '1h'
MARKET_TYPE = 'spot'
DATA_TYPE = 'klines'
ENV_KWARGS = {
    'max_steps': 0,
    'init_balance': 10_000,
    'no_action_finish': np.inf,
    'fee': 0.00075,
    'coin_step': 0.00001,
    # 'verbose': True,
    # 'visualize': True,
}


def main():
    # Wczytanie danych OHLCV
    ohlcv = by_BinanceVision(
        ticker=TICKER,
        interval=ITV,
        market_type=MARKET_TYPE,
        data_type=DATA_TYPE,
        start_date='2018-01-01 00:00:00',
        end_date='2025-01-01 00:00:00',
        split=False,
        delay=1_000_000
    )
    ohlcv['Opened'] = pd.to_datetime(ohlcv['Opened'])
    ohlcv.set_index('Opened', inplace=True)
    print(f'Dane OHLCV załadowane: {ohlcv.shape[0]} wierszy')

    results_list = []

    for filepath in glob.glob(FILE_PATTERN):
        df = pd.read_csv(filepath)
        df['Opened'] = pd.to_datetime(df['Opened'])
        df.set_index('Opened', inplace=True)

        action_sequence = df['action'].tolist()

        # Określenie zakresu czasowego na podstawie danych akcji
        start_time = df.index.min()
        end_time = df.index.max()

        # Wybranie odpowiedniego fragmentu danych OHLCV
        ohlcv_part = ohlcv.loc[start_time:end_time].reset_index()

        env = SignalCleanSpotEnv(df=ohlcv_part, **ENV_KWARGS)
        env.reset()
        env.signals = action_sequence
        env_results = env()

        # env.signals po przejściu przez środowisko
        actions = env.signals

        # Łączymy dane ohlcv_part z actions
        ohlcv_part['Action'] = actions
        ohlcv_part['Weight'] = env.weights

        # Dodajemy do listy wyników
        results_list.append(ohlcv_part)

        print(f"Przetworzono plik: {filepath}")

    # Łączymy wszystkie części w jeden duży DataFrame
    final_df = pd.concat(results_list, ignore_index=True)

    # Upewniamy się, że dane są posortowane według czasu
    final_df.sort_values('Opened', inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    # Zapisujemy finalny DataFrame do jednego pliku CSV
    final_filename = os.path.join(absolute_path, 'final_combined_actions.csv')
    final_df.to_csv(final_filename, index=False)
    print(f"Zapisano połączone dane do: {final_filename}")


if __name__ == '__main__':
    main()
