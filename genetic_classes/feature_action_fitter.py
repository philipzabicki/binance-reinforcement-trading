from numpy import nan_to_num, array
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice
# from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from talib import AD, TRANGE

from utils.feature_generation import *
# from scipy.signal import correlate
# from tslearn.metrics import dtw
from utils.ta_tools import *


class KeltnerChannelFitting(ElementwiseProblem):
    def __init__(self, df, target_segments, min_match_factor=0.01, *args, **kwargs):
        print(f'Class KeltnerChannelFitting df: {df}')
        self.target_segments = target_segments
        self.target_segments_count = len(target_segments)
        self.df = df.iloc[:, 1:].to_numpy().astype(float)
        print(f'Class KeltnerChannelFitting self.df: {self.df}')
        self.min_match_factor = min_match_factor
        bands_variables = {
            "ma_type": Integer(bounds=(0, 31)),
            "atr_ma_type": Integer(bounds=(0, 25)),
            "ma_period": Integer(bounds=(2, 1000)),
            "atr_period": Integer(bounds=(2, 1000)),
            "atr_multi": Real(bounds=(0.001, 15.000)),
            "source": Choice(options=["open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4"])
        }
        super().__init__(*args, vars=bands_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        signals = custom_keltner_channel_signal_cached(
            ohlcv=self.df[:, :5],  # cols 0,1,2,3,4 -> OHLCV
            ma_type=X['ma_type'],
            ma_period=X['ma_period'],
            atr_ma_type=X['atr_ma_type'],
            atr_period=X['atr_period'],
            atr_multi=X['atr_multi'],
            source=X['source']
        )
        signals = np.where(signals >= 1, 1, np.where(signals <= -1, -1, 0))
        extracted_signals = extract_segments_indices(signals)

        # Optymalizacja: konwersja wyekstrahowanych segmentów do zbioru
        extracted_signals_set = {tuple(seg) for seg in extracted_signals}
        match_count = sum(1 for seg in self.target_segments if tuple(seg) in extracted_signals_set)
        nonzero_signals = np.count_nonzero(signals)

        epsilon = 1e-8
        threshold = self.min_match_factor * self.target_segments_count

        if match_count < threshold:
            missing_ratio = (threshold - match_count) / threshold
            penalty = missing_ratio * self.target_segments_count
            out["F"] = penalty
        else:
            ratio1 = match_count / self.target_segments_count
            ratio2 = match_count / (nonzero_signals + epsilon)
            out["F"] = - (ratio1 + ratio2) / 2


class MACDFitting(ElementwiseProblem):
    def __init__(self, df, target_segments, min_match_factor=0.01, *args, **kwargs):
        self.target_segments = target_segments
        self.target_segments_count = len(target_segments)
        self.df = df.iloc[:, 1:].to_numpy().astype(float)
        print(f'Class MACDFitting self.df: {self.df}')
        self.min_match_factor = min_match_factor
        macd_variables = {
            "signal_type": Choice(options=["lines_cross_with_zero", "lines_cross",
                                           "lines_approaching_cross_with_zero", "lines_approaching_cross",
                                           "signal_line_zero_cross", "MACD_line_zero_cross",
                                           "histogram_reversal"]),
            "fast_source": Choice(options=["open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4"]),
            "slow_source": Choice(options=["open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4"]),
            "fast_period": Integer(bounds=(2, 1000)),
            "slow_period": Integer(bounds=(2, 1000)),
            "signal_period": Integer(bounds=(2, 1000)),
            "fast_ma_type": Integer(bounds=(0, 31)),
            "slow_ma_type": Integer(bounds=(0, 31)),
            "signal_ma_type": Integer(bounds=(0, 25))}
        super().__init__(*args, vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        macd, macd_signal = custom_MACD_cached(
            fast_source=X['fast_source'],
            slow_source=X['slow_source'],
            fast_ma_type=X['fast_ma_type'], fast_period=X['fast_period'],
            slow_ma_type=X['slow_ma_type'], slow_period=X['slow_period'],
            signal_ma_type=X['signal_ma_type'], signal_period=X['signal_period']
        )
        signal_func_mapping = {
            'lines_cross_with_zero': MACD_lines_cross_with_zero,
            'lines_cross': MACD_lines_cross,
            'lines_approaching_cross_with_zero': MACD_lines_approaching_cross_with_zero,
            'lines_approaching_cross': MACD_lines_approaching_cross,
            'signal_line_zero_cross': MACD_signal_line_zero_cross,
            'MACD_line_zero_cross': MACD_line_zero_cross,
            'histogram_reversal': MACD_histogram_reversal
        }
        try:
            # Use mapping to select the function based on signals_source
            func = signal_func_mapping[X['signal_type']]
        except KeyError:
            raise ValueError('Unknown signals source, available {lines_cross_with_zero, lines_cross, '
                             'lines_approaching_cross_with_zero, lines_approaching_cross, '
                             'signal_line_zero_cross, MACD_line_zero_cross, histogram_reversal}')

        signals = array(func(macd, macd_signal))
        extracted_signals = extract_segments_indices(signals)
        extracted_signals_set = {tuple(seg) for seg in extracted_signals}
        match_count = sum(1 for seg in self.target_segments if tuple(seg) in extracted_signals_set)
        nonzero_signals = np.count_nonzero(signals)

        epsilon = 1e-8
        threshold = self.min_match_factor * self.target_segments_count

        if match_count < threshold:
            missing_ratio = (threshold - match_count) / threshold
            penalty = missing_ratio * self.target_segments_count
            out["F"] = penalty
        else:
            ratio1 = match_count / self.target_segments_count
            ratio2 = match_count / (nonzero_signals + epsilon)
            out["F"] = - (ratio1 + ratio2) / 2


class StochasticOscillatorFitting(ElementwiseProblem):
    def __init__(self, df, target_segments, min_match_factor=0.01, *args, **kwargs):
        self.target_segments = target_segments
        self.target_segments_count = len(target_segments)
        self.df = df.iloc[:, 1:].to_numpy().astype(float)
        # print(f'Class StochasticOscillatorFitting self.df: {self.df}')
        self.min_match_factor = min_match_factor

        stoch_variables = {
            "signal_type": Choice(options=[
                "k_int_cross", "k_ext_cross", "d_int_cross", "d_ext_cross",
                "k_cross_int_os_ext_ob", "k_cross_ext_os_int_ob",
                "d_cross_int_os_ext_ob", "d_cross_ext_os_int_ob",
                "kd_cross", "kd_cross_inside", "kd_cross_outside"
            ]),
            "fastK_period": Integer(bounds=(2, 250)),
            "slowK_period": Integer(bounds=(1, 250)),
            "slowD_period": Integer(bounds=(2, 250)),
            "slowK_ma_type": Integer(bounds=(0, 25)),
            "slowD_ma_type": Integer(bounds=(0, 25)),
            "oversold_threshold": Real(bounds=(0.0, 50.0)),
            "overbought_threshold": Real(bounds=(50.0, 100.0))
        }
        super().__init__(*args, vars=stoch_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # Obliczenie wartości oscylatora stochastycznego (slowK oraz slowD)
        slowK, slowD = custom_StochasticOscillator_cached(
            fastK_period=X['fastK_period'],
            slowK_period=X['slowK_period'],
            slowD_period=X['slowD_period'],
            slowK_ma_type=X['slowK_ma_type'],
            slowD_ma_type=X['slowD_ma_type']
        )

        # Mapowanie typów sygnałów na odpowiadające im funkcje.
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
            "kd_cross_outside": kd_cross_outside
        }
        signal_type = X['signal_type']
        if signal_type not in signal_func_mapping:
            raise ValueError("Unknown signal type. Available options: " +
                             ", ".join(signal_func_mapping.keys()))
        func = signal_func_mapping[signal_type]

        # Wywołanie wybranej funkcji sygnałowej z przekazaniem dodatkowych argumentów.
        signals = np.array(func(k_line=slowK, d_line=slowD, oversold_threshold=X['oversold_threshold'],
                                overbought_threshold=X['overbought_threshold']))

        # Obliczenie metryki, analogicznie do klasy MACDFitting.
        extracted_signals = extract_segments_indices(signals)
        extracted_signals_set = {tuple(seg) for seg in extracted_signals}
        match_count = sum(1 for seg in self.target_segments if tuple(seg) in extracted_signals_set)
        nonzero_signals = np.count_nonzero(signals)

        epsilon = 1e-8
        threshold = self.min_match_factor * self.target_segments_count

        if match_count < threshold:
            missing_ratio = (threshold - match_count) / threshold
            penalty = missing_ratio * self.target_segments_count
            out["F"] = penalty
        else:
            ratio1 = match_count / self.target_segments_count
            ratio2 = match_count / (nonzero_signals + epsilon)
            out["F"] = - (ratio1 + ratio2) / 2


class KeltnerChannelFitting_v1(ElementwiseProblem):
    def __init__(self, df, lower, upper, ma_type, *args, **kwargs):
        self.ma_type = ma_type
        self.actions = df['Action'].values
        weights = df['Weight'].values
        self.mask = (weights > lower) & (weights <= upper)
        self.df = df.to_numpy()
        self.trange = TRANGE(*self.df[:, 1:4].T.astype(float))
        bands_variables = {
            "ma_period": Integer(bounds=(2, 1000)),
            "atr_ma_type": Integer(bounds=(0, 25)),
            "atr_period": Integer(bounds=(2, 1000)),
            "atr_multi": Real(bounds=(0.001, 15.000)),
            "source": Choice(options=["open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4"])
        }
        super().__init__(*args, vars=bands_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        signals = custom_keltner_channel_signal(
            np_df=self.df[:, 1:6].astype(float),
            true_range=self.trange,
            ma_type=self.ma_type,
            ma_period=X['ma_period'],
            atr_ma_type=X['atr_ma_type'],
            atr_period=X['atr_period'],
            atr_multi=X['atr_multi'],
            source=X['source']
        )
        # signals = where(signals >= 1, 1, where(signals <= -1, -1, 0))

        # out["F"] = sqrt(mean((nan_to_num(signals_masked) - actions_masked) ** 2))
        # out["F"] = euclidean(nan_to_num(signals_masked), actions_masked)
        out["F"] = mean_squared_error(nan_to_num(signals[self.mask]), self.actions[self.mask])


class MACDFitting_v1(ElementwiseProblem):
    def __init__(self, df, lower, upper, *args, signals_source='cross', **kwargs):
        self.actions = df['Action'].values
        weights = df['Weight'].values
        self.mask = (weights > lower) & (weights <= upper)
        self.df = df.to_numpy()
        macd_variables = {
            "fast_source": Choice(options=["open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4"]),
            "slow_source": Choice(options=["open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4"]),
            "fast_period": Integer(bounds=(2, 1000)),
            "slow_period": Integer(bounds=(2, 1000)),
            "signal_period": Integer(bounds=(2, 1000)),
            "fast_ma_type": Integer(bounds=(0, 31)),
            "slow_ma_type": Integer(bounds=(0, 31)),
            "signal_ma_type": Integer(bounds=(0, 25))}
        self.signals_source = signals_source
        super().__init__(*args, vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        macd, macd_signal = custom_MACD(self.df[:, 1:6].astype(float),
                                        fast_source=X['fast_source'],
                                        slow_source=X['slow_source'],
                                        fast_ma_type=X['fast_ma_type'], fast_period=X['fast_period'],
                                        slow_ma_type=X['slow_ma_type'], slow_period=X['slow_period'],
                                        signal_ma_type=X['signal_ma_type'], signal_period=X['signal_period'])
        if self.signals_source == 'cross':
            signals = array(MACD_cross_signal(macd, macd_signal))
        elif self.signals_source == 'zero-cross':
            signals = array(MACD_zero_cross_signal(macd, macd_signal))
        elif self.signals_source == 'histogram-reversal':
            signals = array(MACD_histogram_reversal_signal(macd - macd_signal))
        else:
            raise ValueError('Unknown signals source, available {cross, zero-cross, histogram-reversal} ')
        # signals = where(signals >= 1, 1, where(signals <= -1, -1, 0))

        # out["F"] = euclidean(nan_to_num(signals[mask]), self.actions[mask])
        out["F"] = mean_squared_error(nan_to_num(signals[self.mask]), self.actions[self.mask])


class StochasticOscillatorFitting_v1(ElementwiseProblem):
    def __init__(self, df, lower, upper, *args, signals_source='cross', **kwargs):
        self.signals_source = signals_source
        self.actions = df['Action'].values
        weights = df['Weight'].values
        self.mask = (weights > lower) & (weights <= upper)
        self.df = df.to_numpy()
        stoch_variables = {
            "fastK_period": Integer(bounds=(2, 250)),
            "slowK_period": Integer(bounds=(1, 250)),
            "slowD_period": Integer(bounds=(2, 250)),
            "slowK_ma_type": Integer(bounds=(0, 25)),
            "slowD_ma_type": Integer(bounds=(0, 25))
        }
        if self.signals_source == 'mid-cross':
            stoch_variables.update({"mid_level": Real(bounds=(0.0, 100.0))})
        elif self.signals_source in ['cross', 'threshold']:
            stoch_variables.update({"oversold_threshold": Real(bounds=(0.0, 50.0)),
                                    "overbought_threshold": Real(bounds=(50.0, 100.0))})
        else:
            raise ValueError('Unknown signals source, available {cross, threshold, mid-cross} ')
        super().__init__(*args, vars=stoch_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        slowK, slowD = custom_StochasticOscillator(self.df[:, 1:6],
                                                   fastK_period=X['fastK_period'],
                                                   slowK_period=X['slowK_period'],
                                                   slowD_period=X['slowD_period'],
                                                   slowK_ma_type=X['slowK_ma_type'],
                                                   slowD_ma_type=X['slowD_ma_type'])
        # print(f'slowK quantiles: {percentile(slowK, [0, 50, 100])}')
        # print(f'slowD quantiles: {percentile(slowD, [0, 50, 100])}', end='\n')
        if self.signals_source == 'cross':
            signals = array(StochasticOscillator_threshold_cross_signal(slowK,
                                                                        slowD,
                                                                        oversold_threshold=X['oversold_threshold'],
                                                                        overbought_threshold=X['overbought_threshold']))
        elif self.signals_source == 'threshold':
            signals = array(StochasticOscillator_threshold_signal(slowK,
                                                                  slowD,
                                                                  oversold_threshold=X['oversold_threshold'],
                                                                  overbought_threshold=X['overbought_threshold']))
        elif self.signals_source == 'mid-cross':
            signals = array(StochasticOscillator_mid_cross_signal(slowK,
                                                                  slowD,
                                                                  mid_level=X['mid_level']))
        else:
            raise ValueError('Unknown signals source, available {cross, threshold, mid_cross} ')
        # signals = where(signals >= 1, 1, where(signals <= -1, -1, 0))
        # out["F"] = euclidean(nan_to_num(signals[mask]), self.actions[mask])
        out["F"] = mean_squared_error(nan_to_num(signals[self.mask]), self.actions[self.mask])


class ADXFitting(ElementwiseProblem):
    def __init__(self, df, lower, upper, *args, signals_source='cross', **kwargs):
        self.signals_source = signals_source
        self.actions = df['Action'].values
        weights = df['Weight'].values
        self.mask = (weights > lower) & (weights <= upper)
        self.df = df.to_numpy()
        self.trange = np.nan_to_num(TRANGE(*self.df[:, 2:5].T.astype(float)))
        print(f'TRANGE {np.quantile(self.trange, [0.25, 0.5, 0.75])}')
        stoch_variables = {
            "atr_period": Integer(bounds=(2, 250)),
            "posDM_period": Integer(bounds=(2, 250)),
            "negDM_period": Integer(bounds=(2, 250)),
            "adx_period": Integer(bounds=(2, 250)),
            "ma_type_atr": Integer(bounds=(0, 25)),
            "ma_type_posDM": Integer(bounds=(0, 25)),
            "ma_type_negDM": Integer(bounds=(0, 25)),
            "ma_type_adx": Integer(bounds=(0, 25)),
        }
        super().__init__(*args, vars=stoch_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        adx, plus_DI, minus_DI = custom_ADX(ohlcv=self.df[:, 1:6],
                                            true_range=self.trange,
                                            atr_period=X['atr_period'],
                                            posDM_period=X['posDM_period'],
                                            negDM_period=X['negDM_period'],
                                            adx_period=X['adx_period'],
                                            ma_type_atr=X['ma_type_atr'],
                                            ma_type_posDM=X['ma_type_posDM'],
                                            ma_type_negDM=X['ma_type_negDM'],
                                            ma_type_adx=X['ma_type_adx']
                                            )
        # print(f'slowK quantiles: {percentile(slowK, [0, 50, 100])}')
        # print(f'slowD quantiles: {percentile(slowD, [0, 50, 100])}', end='\n')
        if self.signals_source == 'cross':
            signals = array(ADX_signal(adx_col=adx,
                                       minus_di=minus_DI,
                                       plus_di=plus_DI))
        elif self.signals_source == 'trend':
            signals = array(ADX_trend_signal(adx_col=adx,
                                             minus_di=minus_DI,
                                             plus_di=plus_DI))
        else:
            raise ValueError('Unknown signals source, available {cross, trend}')
        # signals = where(signals >= 1, 1, where(signals <= -1, -1, 0))
        # out["F"] = euclidean(nan_to_num(signals[mask]), self.actions[mask])
        # time.sleep(60)
        out["F"] = mean_squared_error(nan_to_num(signals[self.mask]), self.actions[self.mask])


class ChaikinOscillatorFitting(ElementwiseProblem):
    def __init__(self, df, lower, upper, *args, **kwargs):
        self.actions = df['Action'].values
        weights = df['Weight'].values
        self.mask = (weights > lower) & (weights <= upper)

        # Oblicz ADL
        self.adl = AD(df['High'].values, df['Low'].values, df['Close'].values, df['Volume'].values)

        chaikin_variables = {"fast_period": Integer(bounds=(2, 500)),
                             "slow_period": Integer(bounds=(2, 500)),
                             "fast_ma_type": Integer(bounds=(0, 25)),
                             "slow_ma_type": Integer(bounds=(0, 25))}
        super().__init__(*args, vars=chaikin_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        chaikin_oscillator = custom_ChaikinOscillator(self.adl,
                                                      fast_ma_type=X['fast_ma_type'], fast_period=X['fast_period'],
                                                      slow_ma_type=X['slow_ma_type'], slow_period=X['slow_period'])
        signals = array(ChaikinOscillator_signal(chaikin_oscillator))
        # signals = where(signals >= 1, 1, where(signals <= -1, -1, 0))
        # out["F"] = euclidean(nan_to_num(signals[mask]), self.actions[mask])
        out["F"] = mean_squared_error(nan_to_num(signals[self.mask]), self.actions[self.mask])

# class KeltnerChannelFitting(ElementwiseProblem):
#     def __init__(self, df, *args, **kwargs):
#         self.weighted_actions = df['Action']*df['Weight']
#         print(f'df {df}')
#         self.df = df.to_numpy()
#         bands_variables = {"atr_period": Integer(bounds=(2, 500)),
#                            "ma_type": Integer(bounds=(0, 0)),
#                            "ma_period": Integer(bounds=(2, 500)),
#                            "atr_multi": Real(bounds=(0.001, 15.000))}
#         super().__init__(*args, vars=bands_variables, n_obj=1, **kwargs)
#
#     def _evaluate(self, X, out, *args, **kwargs):
#         # print(f'X: {X}')
#         signals = nan_to_num(get_MA_band_signal(self.df[:, 1:6].astype(float),
#                                      X['ma_type'], X['ma_period'],
#                                      X['atr_period'], X['atr_multi']))
#         # print(f'self.weighted_actions {self.weighted_actions}')
#         # print(f'signals {signals}')
#         signals = where(signals >= 1, 1, where(signals <= -1, -1, 0))
#         # out["F"] = 1 - spearmanr(signals, self.weighted_actions)[0]
#         # r, _ = spearmanr(signals, self.weighted_actions)
#         # score = 1 - r
#         out["F"] = euclidean(signals*self.df[:,-1], self.weighted_actions)
#         # out["F"] = sqrt(mean((signals - self.weighted_actions) ** 2))

# class KeltnerChannelVarSourceFitting(ElementwiseProblem):
#     def __init__(self, df, lower, upper, ma_type, *args, **kwargs):
#         self.ma_type = ma_type
#         df.loc[abs(df['Weight']) < lower, ['Weight', 'Action']] = [0.0, 0]
#         df.loc[abs(df['Weight']) > upper, ['Weight', 'Action']] = [0.0, 0]
#         self.actions = df['Action']
#         self.weights = df['Weight']
#         self.weighted_actions = df['Action']*df['Weight']
#         self.df = df.to_numpy()
#         bands_variables = {"atr_period": Integer(bounds=(2, 1_000)),
#                            # "ma_type": Integer(bounds=(0, 31)),
#                            "ma_period": Integer(bounds=(2, 1_000)),
#                            "atr_multi": Real(bounds=(0.001, 15.000)),
#                            "source": Choice(options=["open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4"])
#                            }
#         super().__init__(*args, vars=bands_variables, n_obj=1, **kwargs)
#
#     def _evaluate(self, X, out, *args, **kwargs):
#         # print(f'X: {X}')
#         signals = get_ma_band_signal_by_source(self.df[:, 1:6].astype(float),
#                                      self.ma_type, X['ma_period'],
#                                      X['atr_period'], X['atr_multi'],
#                                                X['source'])
#         signals = where(signals >= 1, 1, where(signals <= -1, -1, 0))
#         print(signals)
#         # out["F"] = 1 - spearmanr(signals, self.weighted_actions)[0]
#         # out["F"] = sqrt(mean((signals*self.weights - self.weighted_actions) ** 2))
#         out["F"] = euclidean(nan_to_num(signals) * self.weights, self.weighted_actions)
