import os

import mplfinance as mpf
import numpy as np
import pandas as pd

from definitions import REPORT_DIR
from utils.ta_tools import MACD_lines_cross_with_zero, MACD_lines_cross, MACD_lines_approaching_cross_with_zero, \
    MACD_lines_approaching_cross, MACD_signal_line_zero_cross, MACD_line_zero_cross, MACD_histogram_reversal
from utils.ta_tools import get_ma_from_source, get_1D_MA

ACTIONS_FULLPATH = os.path.join(
    REPORT_DIR, "optimal_actions", "final_combined_actions.csv"
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
settings = {'signal_type': 'lines_cross', 'fast_source': 'open', 'slow_source': 'open', 'fast_period': 975,
            'slow_period': 3, 'signal_period': 3, 'fast_ma_type': 5, 'slow_ma_type': 29, 'signal_ma_type': 9}


def main():
    # Wczytanie danych i ustawienie indeksu czasowego
    d_olhcv_aw = pd.read_csv(ACTIONS_FULLPATH)
    d_olhcv_aw['Opened'] = pd.to_datetime(d_olhcv_aw['Opened'])
    d_olhcv_aw.set_index('Opened', inplace=True)
    d_olhcv_aw.sort_index(inplace=True)

    # Konwersja do numpy array do obliczeń
    np_olhcv = d_olhcv_aw[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy().astype(float)

    # Obliczanie linii MACD oraz linii sygnału
    macd_line = get_ma_from_source(
        np_olhcv, settings['fast_ma_type'], settings['fast_period'], settings['fast_source']
    ) - get_ma_from_source(
        np_olhcv, settings['slow_ma_type'], settings['slow_period'], settings['slow_source']
    )
    signal_line = get_1D_MA(macd_line, settings['signal_ma_type'], settings['signal_period'])

    # Dodanie MACD oraz Signal do DataFrame
    d_olhcv_aw['MACD'] = macd_line
    d_olhcv_aw['Signal'] = signal_line

    # Obliczanie sygnałów na podstawie MACD (różnych od sygnałów z kolumny Action)
    signals = signal_func_mapping[settings['signal_type']](macd_line, signal_line)
    # signals = np.where(signals >= 1, 1, np.where(signals <= -1, -1, 0))
    d_olhcv_aw['MACD_signal'] = signals

    # Przygotowanie markerów dla sygnałów z kolumny Action (do wykresu świecowego)
    # Te mają być lekko przezroczyste (alpha=0.5)
    d_olhcv_aw['Action_buy_marker'] = np.nan
    d_olhcv_aw['Action_sell_marker'] = np.nan
    action_buy_idx = d_olhcv_aw['Action'] == 1
    action_sell_idx = d_olhcv_aw['Action'] == -1
    d_olhcv_aw.loc[action_buy_idx, 'Action_buy_marker'] = d_olhcv_aw.loc[action_buy_idx, 'Low'] * 0.995
    d_olhcv_aw.loc[action_sell_idx, 'Action_sell_marker'] = d_olhcv_aw.loc[action_sell_idx, 'High'] * 1.005

    # Przygotowanie markerów dla sygnałów wyliczonych z MACD (do wykresu świecowego)
    # Używamy innych przesunięć, aby nie nakładały się na markery z Action
    d_olhcv_aw['MACD_buy_marker'] = np.nan
    d_olhcv_aw['MACD_sell_marker'] = np.nan
    macd_buy_idx = d_olhcv_aw['MACD_signal'] == 1
    macd_sell_idx = d_olhcv_aw['MACD_signal'] == -1
    d_olhcv_aw.loc[macd_buy_idx, 'MACD_buy_marker'] = d_olhcv_aw.loc[macd_buy_idx, 'Low'] * 0.98
    d_olhcv_aw.loc[macd_sell_idx, 'MACD_sell_marker'] = d_olhcv_aw.loc[macd_sell_idx, 'High'] * 1.02

    # Pobranie próbki 100 kolejnych wierszy do wykresu
    total_rows = len(d_olhcv_aw)
    if total_rows > 100:
        start_idx = np.random.randint(0, total_rows - 100)
        d_sample = d_olhcv_aw.iloc[start_idx:start_idx + 100].copy()
    else:
        d_sample = d_olhcv_aw.copy()

    # Przygotowanie dodatkowych obiektów addplot:
    apds = [
        # Sygnały z kolumny Action na wykresie świecowym (panel 0) z przezroczystością
        mpf.make_addplot(d_sample['Action_buy_marker'], panel=0, type='scatter', markersize=100, marker='^',
                         color='green', alpha=0.5),
        mpf.make_addplot(d_sample['Action_sell_marker'], panel=0, type='scatter', markersize=100, marker='v',
                         color='red', alpha=0.5),
        # Sygnały z MACD na wykresie świecowym (panel 0) – w pełnej przezroczystości
        mpf.make_addplot(d_sample['MACD_buy_marker'], panel=0, type='scatter', markersize=100, marker='^',
                         color='green'),
        mpf.make_addplot(d_sample['MACD_sell_marker'], panel=0, type='scatter', markersize=100, marker='v',
                         color='red'),
        # Linie MACD oraz linii Signal na osobnym panelu (panel 1)
        mpf.make_addplot(d_sample['MACD'], panel=1, color='blue', title='MACD'),
        mpf.make_addplot(d_sample['Signal'], panel=1, color='orange')
    ]

    # Rysowanie wykresu:
    # Panel 0: Wykres świecowy + sygnały (Action i MACD) na poziomie cen
    # Panel 1: Linie MACD oraz Signal
    # Panel 2: Wolumen (dzięki volume_panel=2)
    mpf.plot(
        d_sample,
        type='candle',
        addplot=apds,
        style='charles',
        title='OHLCV Chart with Action & MACD Signals, MACD and Volume (100 rows)',
        volume=True,
        volume_panel=2,
        panel_ratios=(3, 1, 1)
    )


if __name__ == "__main__":
    main()
