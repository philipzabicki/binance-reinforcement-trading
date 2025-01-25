from numpy import nan_to_num, array
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice
# from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from talib import AD, TRANGE

from utils.feature_generation import custom_StochasticOscillator, custom_ChaikinOscillator, custom_MACD, \
    custom_keltner_channel_signal
# from scipy.signal import correlate
# from tslearn.metrics import dtw
from utils.ta_tools import MACD_cross_signal, MACD_histogram_reversal_signal, MACD_zero_cross_signal, \
    StochasticOscillator_signal, ChaikinOscillator_signal


class KeltnerChannelFitting(ElementwiseProblem):
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


class MACDCFitting(ElementwiseProblem):
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


class StochasticOscillatorFitting(ElementwiseProblem):
    def __init__(self, df, lower, upper, *args, **kwargs):
        self.actions = df['Action'].values
        weights = df['Weight'].values
        self.mask = (weights > lower) & (weights <= upper)
        self.df = df.to_numpy()
        stoch_variables = {"oversold_threshold": Real(bounds=(0.0, 50.0)),
                           "overbought_threshold": Real(bounds=(50.0, 100.0)),
                           "fastK_period": Integer(bounds=(2, 250)),
                           "slowK_period": Integer(bounds=(1, 250)),
                           "slowD_period": Integer(bounds=(2, 250)),
                           "slowK_ma_type": Integer(bounds=(0, 25)),
                           "slowD_ma_type": Integer(bounds=(0, 25))
                           }
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
        signals = array(StochasticOscillator_signal(slowK,
                                                    slowD,
                                                    oversold_threshold=X['oversold_threshold'],
                                                    overbought_threshold=X['overbought_threshold']))
        # signals = where(signals >= 1, 1, where(signals <= -1, -1, 0))
        # out["F"] = euclidean(nan_to_num(signals[mask]), self.actions[mask])
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
