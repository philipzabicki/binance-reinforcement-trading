import numpy as np
import pandas as pd
import talib
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from statsmodels.nonparametric.kde import KDEUnivariate

from utils.ta_tools import get_MA_band_signal


def keltner_channels_custom_mas(df, ma_periods, atr_periods, atr_multipliers):
    """
    eg. arguments
    ma_periods = [5, 10, 25, 50, 75, 100]
    atr_perods = [5, 10, 15, 20, 25, 30, 40, 50]
    atr_multipliers = [.25, .5, 1., 1.25, 1.5, 1.75, 2., 2.5, 3.]
    res is 16_416 columns
    """
    # taking only ohlc values and converting to nump for speed
    np_df = df.iloc[:, 1:6].to_numpy()
    print(np_df)
    for ma_p in ma_periods:
        for atr_p in atr_periods:
            for atr_m in atr_multipliers:
                for ma_id in range(0, 38):
                    df[f'{ma_id}MAp{ma_p}atr_p{atr_p}atr_m{atr_m}'] = get_MA_band_signal(np_df, ma_id, ma_p, atr_p,
                                                                                         atr_m)
    return df


def add_scaled_ultosc_rsi_mfi_up_to_n(df, n, step=1):
    indicator_columns = []
    new_columns = {}
    for p in range(2, n + 1, step):
        rsi_column = f'RSI{p}'
        new_columns[rsi_column] = talib.RSI(df['Close'], timeperiod=p)
        indicator_columns.append(rsi_column)
    for p in range(2, n + 1, step):
        mfi_column = f'MFI{p}'
        new_columns[mfi_column] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=p)
        indicator_columns.append(mfi_column)
    for p in range(1, n + 1, step):
        ultosc_column = f'ULTOSC{p}'
        new_columns[ultosc_column] = talib.ULTOSC(df['High'], df['Low'], df['Close'],
                                                  timeperiod1=p, timeperiod2=2 * p, timeperiod3=3 * p)
        indicator_columns.append(ultosc_column)
    new_columns_df = pd.DataFrame(new_columns, index=df.index)
    df = pd.concat([df, new_columns_df], axis=1)
    df[indicator_columns] = df[indicator_columns].fillna(0)
    if indicator_columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[indicator_columns] = scaler.fit_transform(df[indicator_columns])
    return df


def add_candle_patterns_indicators(df):
    candle_names = talib.get_function_groups()['Pattern Recognition']
    for pattern_name in candle_names:
        pattern_function = getattr(talib, pattern_name)
        df[pattern_name] = pattern_function(df['Open'], df['High'], df['Low'], df['Close'])
    return df


def add_scaled_candle_patterns_indicators(df):
    candle_names = talib.get_function_groups().get('Pattern Recognition', [])
    pattern_columns = []
    new_columns = {}
    for pattern_name in candle_names:
        pattern_function = getattr(talib, pattern_name)
        new_columns[pattern_name] = pattern_function(df['Open'], df['High'], df['Low'], df['Close'])
        pattern_columns.append(pattern_name)
    new_columns_df = pd.DataFrame(new_columns, index=df.index)
    df = pd.concat([df, new_columns_df], axis=1)
    if pattern_columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[pattern_columns] = scaler.fit_transform(df[pattern_columns])
    return df


def change_kde_pdf(df, column='Close', start_index=10, cpus=-1):
    def _compute_prob_t(t, current_diff, increases_up_to_t, decreases_up_to_t):
        prob = 0.0
        if current_diff > 0:
            kde_increase = gaussian_kde(increases_up_to_t[~np.isnan(increases_up_to_t)])
            prob = kde_increase.integrate_box_1d(-np.inf, current_diff)
        elif current_diff < 0:
            kde_decrease = gaussian_kde(decreases_up_to_t[~np.isnan(decreases_up_to_t)])
            prob = kde_decrease.integrate_box_1d(current_diff, np.inf)
        return prob

    diffs = np.ascontiguousarray(df[column].diff().values)
    increases = np.ascontiguousarray(np.where(diffs >= 0, diffs, np.nan))
    decreases = np.ascontiguousarray(np.where(diffs <= 0, diffs, np.nan))
    probs_list = Parallel(n_jobs=cpus, backend='threading')(
        delayed(_compute_prob_t)(t, diffs[t], increases[:t], decreases[:t]) for t in range(start_index, diffs.shape[0])
    )
    df[f'{column}_change_KDE_gauss'] = np.concatenate((np.zeros(start_index), probs_list))
    return df


def candle_sizes_kde_pdf(df, start_index=10, cpus=-1):
    def _compute_prob_candle(t, vals_up_to_t):
        return gaussian_kde(vals_up_to_t).integrate_box_1d(0, vals_up_to_t[t])

    df['candle_size'] = df['High'] - df['Low']
    df['candle_size'] = df['candle_size'].replace(0, df['candle_size'].median())
    df['candle_body_size'] = np.absolute(df['Close'] - df['Open'])
    df['candle_upshadow_size'] = (df['High'] - df[['Open', 'Close']].max(axis=1))
    df['candle_downshadow_size'] = (df[['Open', 'Close']].min(axis=1) - df['Low'])
    df['candle_body_ratio'] = df['candle_body_size'] / df['candle_size']
    df['candle_upshadow_ratio'] = df['candle_upshadow_size'] / df['candle_size']
    df['candle_downshadow_ratio'] = df['candle_downshadow_size'] / df['candle_size']

    for f in ['candle_size', 'candle_body_size', 'candle_upshadow_size', 'candle_downshadow_size', 'candle_body_ratio',
              'candle_upshadow_ratio', 'candle_downshadow_ratio']:
        candle_part = np.ascontiguousarray(df[f].values)
        probs_list = Parallel(n_jobs=cpus, backend='threading')(
            delayed(_compute_prob_candle)(t, candle_part[:t + 1]) for t in range(start_index, df.shape[0])
        )
        df[f'{f}_kde_gauss_pdf'] = np.concatenate((np.zeros(start_index), probs_list))

    df.drop(columns=['candle_size', 'candle_body_size', 'candle_upshadow_size', 'candle_downshadow_size'], inplace=True)
    return df


def candle_sizes_kde_pdf_kernels(df, kernels=['gau'], start=10, cpus=-1):
    def _compute_prob_candle_kernel(t, vals_up_to_t, kernel='gau'):
        kde = KDEUnivariate(vals_up_to_t)
        kde.fit(kernel=kernel, bw='silverman', fft=False)
        grid_points = kde.support
        cdf_values = kde.cdf
        # Interpolate the CDF at vals_up_to_t[t]
        cdf_interp = interp1d(grid_points, cdf_values, bounds_error=False, fill_value=(0, 1))
        cdf_value = cdf_interp(vals_up_to_t[t])
        return cdf_value

    df['candle_size'] = df['High'] - df['Low']
    df['candle_size'] = df['candle_size'].replace(0, df['candle_size'].median())
    df['candle_body_size'] = np.absolute(df['Close'] - df['Open'])
    df['candle_upshadow_size'] = (df['High'] - df[['Open', 'Close']].max(axis=1))
    df['candle_downshadow_size'] = (df[['Open', 'Close']].min(axis=1) - df['Low'])
    df['candle_body_ratio'] = df['candle_body_size'] / df['candle_size']
    df['candle_upshadow_ratio'] = df['candle_upshadow_size'] / df['candle_size']
    df['candle_downshadow_ratio'] = df['candle_downshadow_size'] / df['candle_size']

    for kernel in kernels:
        for f in ['candle_size', 'candle_body_size', 'candle_upshadow_size', 'candle_downshadow_size',
                  'candle_body_ratio', 'candle_upshadow_ratio', 'candle_downshadow_ratio']:
            candle_part = np.ascontiguousarray(df[f].values)
            probs = np.ascontiguousarray(np.zeros(df.shape[0]))
            probs_list = Parallel(n_jobs=cpus, backend='threading')(
                delayed(_compute_prob_candle_kernel)(t, candle_part[:t + 1], kernel=kernel) for t in
                range(start, df.shape[0])
            )
            probs[start:] = probs_list
            df[f'{f}_kde_{kernel}_pdf'] = probs
    return df

# def _compute_prob_t(t, diffs, increases, decreases):
#     prob = 0.0
#     increases_up_to_t = increases[:t]
#     decreases_up_to_t = decreases[:t]
#
#     if diffs[t] > 0:
#         kde_increase = gaussian_kde(increases_up_to_t[~np.isnan(increases_up_to_t)])
#         prob = kde_increase.integrate_box_1d(0, diffs[t])
#     elif diffs[t] < 0:
#         kde_decrease = gaussian_kde(decreases_up_to_t[~np.isnan(decreases_up_to_t)])
#         prob = kde_decrease.integrate_box_1d(diffs[t], 0)
#     return prob
#
# def compute_probabilities_kde_separate_parallel(df, start=10, cpus=-1):
#     # df = df.copy()
#     df['CO_ratio'] = (df['Close'] / df['Open']) - 1
#     diffs = np.ascontiguousarray(df['CO_ratio'].values)
#     N = len(diffs)
#     probs = np.ascontiguousarray(np.zeros(N))
#     increases = np.ascontiguousarray(np.where(diffs > 0, diffs, np.nan))
#     decreases = np.ascontiguousarray(np.where(diffs < 0, diffs, np.nan))
#
#     # Użycie backendu 'threading' dla równoległości wątkowej
#     probs_list = Parallel(n_jobs=cpus, backend='threading')(
#         delayed(_compute_prob_t)(t, diffs, increases, decreases) for t in range(start, N)
#     )
#     probs[start:] = probs_list
#
#     df['CO_ratio_KDE_gauss'] = probs
#     return df

# def compute_probabilities_kde(df):
#     df = df.copy()
#     df['CO_ratio'] = (df['Close']/df['Open'])-1
#     diffs = df['CO_ratio'].values
#     N = len(diffs)
#     probs = np.zeros(N)
#
#     for t in range(2, N):
#         # print(f'{t/N}')
#         probs[t] = gaussian_kde(diffs[:t]).integrate_box_1d(-np.inf, diffs[t])
#         # print(probs[t-2:t+2])
#     df['prob'] = probs
#     return df
