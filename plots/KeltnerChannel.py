import os

import mplfinance as mpf
import numpy as np
import pandas as pd
from talib import TRANGE

from definitions import REPORT_DIR
from utils.ta_tools import get_ma_from_source, get_1D_MA

ACTIONS_FULLPATH = os.path.join(
    REPORT_DIR, "optimal_actions", "final_combined_actions.csv"
)
settings = {'ma_type': 20, 'atr_ma_type': 21, 'ma_period': 8, 'atr_period': 9, 'atr_multi': 0.6115465660842656,
            'source': 'low'}


def main():
    # Read full CSV data
    d_olhcv_aw = pd.read_csv(ACTIONS_FULLPATH)

    # Convert 'Opened' column to datetime and set it as index
    d_olhcv_aw['Opened'] = pd.to_datetime(d_olhcv_aw['Opened'])
    d_olhcv_aw.set_index('Opened', inplace=True)
    d_olhcv_aw.sort_index(inplace=True)  # Ensure data is chronologically ordered

    # Extract OHLCV values from the full dataset
    np_olhcv = d_olhcv_aw[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy().astype(float)

    # Compute technical indicators on the full dataset
    true_range = TRANGE(np_olhcv[:, 1], np_olhcv[:, 2], np_olhcv[:, 3])
    atr = get_1D_MA(true_range, settings['atr_ma_type'], settings['atr_period'])
    ma = get_ma_from_source(np_olhcv, settings['ma_type'], settings['ma_period'], settings['source'])

    # Calculate Keltner Channel bands using the full dataset
    d_olhcv_aw['ma'] = ma
    d_olhcv_aw['upper_band'] = ma + atr * settings['atr_multi']
    d_olhcv_aw['lower_band'] = ma - atr * settings['atr_multi']

    # Create marker columns for optimal buy and sell actions
    d_olhcv_aw['buy_marker'] = np.nan
    d_olhcv_aw['sell_marker'] = np.nan
    buy_idx = d_olhcv_aw['Action'] == 1
    sell_idx = d_olhcv_aw['Action'] == -1

    # Position markers slightly below Low for buys and above High for sells
    d_olhcv_aw.loc[buy_idx, 'buy_marker'] = d_olhcv_aw.loc[buy_idx, 'Low'] * 0.995
    d_olhcv_aw.loc[sell_idx, 'sell_marker'] = d_olhcv_aw.loc[sell_idx, 'High'] * 1.005

    # Sample 1000 consecutive rows from the full dataset (with precomputed indicators)
    total_rows = len(d_olhcv_aw)
    if total_rows > 100:
        start_idx = np.random.randint(0, total_rows - 100)
        d_sample = d_olhcv_aw.iloc[start_idx:start_idx + 100].copy()
    else:
        d_sample = d_olhcv_aw.copy()

    # Prepare additional plots: moving average, upper and lower bands, and action markers
    apds = [
        mpf.make_addplot(d_sample['ma'], color='blue'),
        mpf.make_addplot(d_sample['upper_band'], color='green'),
        mpf.make_addplot(d_sample['lower_band'], color='red'),
        mpf.make_addplot(d_sample['buy_marker'], type='scatter', markersize=100, marker='^', color='green'),
        mpf.make_addplot(d_sample['sell_marker'], type='scatter', markersize=100, marker='v', color='red'),
    ]

    # Plot the candlestick chart including volume along with the Keltner channel and optimal actions
    mpf.plot(d_sample, type='candle', addplot=apds, style='charles',
             title='OHLCV Chart with Keltner Channel and Optimal Actions (1000 consecutive rows)', volume=True)


if __name__ == "__main__":
    main()
