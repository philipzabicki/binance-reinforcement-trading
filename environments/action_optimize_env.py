from numpy import int32, ones
from numpy.random import choice

from .base import SpotBacktest


class SignalCleanSpotEnv(SpotBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "position_ratio" in kwargs:
            self.position_ratio = (
                kwargs["position_ratio"] / 100
                if isinstance(kwargs["position_ratio"], int32)
                else kwargs["position_ratio"]
            )
        else:
            self.position_ratio = 1.0
        if "save_ratio" in kwargs:
            self.save_ratio = (
                kwargs["save_ratio"] / 100
                if isinstance(kwargs["save_ratio"], int32)
                else kwargs["save_ratio"]
            )
        else:
            self.save_ratio = None
        self.stop_loss = kwargs["stop_loss"] if "stop_loss" in kwargs else None
        self.take_profit = kwargs["take_profit"] if "take_profit" in kwargs else None
        self.enter_threshold = kwargs["enter_at"] if "enter_at" in kwargs else 1.0
        self.close_threshold = kwargs["close_at"] if "close_at" in kwargs else 1.0
        # self.signals = empty(self.total_steps)
        self.signals = choice([-1, 0, 1], size=self.total_steps)
        self.weights = ones(self.total_steps, dtype=float)

    def reset(self, *args, **kwargs):
        if "position_ratio" in kwargs:
            self.position_ratio = (
                kwargs["position_ratio"] / 100
                if isinstance(kwargs["position_ratio"], int32)
                else kwargs["position_ratio"]
            )
        else:
            self.position_ratio = 1.0
        if "save_ratio" in kwargs:
            self.save_ratio = (
                kwargs["save_ratio"] / 100
                if isinstance(kwargs["save_ratio"], int32)
                else kwargs["save_ratio"]
            )
        else:
            self.save_ratio = None
        self.stop_loss = kwargs["stop_loss"] if "stop_loss" in kwargs else None
        self.take_profit = kwargs["take_profit"] if "take_profit" in kwargs else None
        self.enter_threshold = kwargs["enter_at"] if "enter_at" in kwargs else 1.0
        self.close_threshold = kwargs["close_at"] if "close_at" in kwargs else 1.0
        return super().reset()

    def _finish_episode(self):
        super()._finish_episode()
        sig_sum = sum(self.signals)
        print(f"Pre sum of signals: {sum(self.signals)}")
        if sig_sum == -1:
            self.signals[-1] = 0
        elif sig_sum == 1 and self.signals[-1] == 1:
            self.signals[-1] = 0
        elif sig_sum == 1:
            self.signals[-1] = -1
        print(f"Post sum of signals: {sum(self.signals)}")
        if self.verbose:
            print(f" position_ratio={self.position_ratio:.2f}")
            if (self.take_profit is not None) and (self.stop_loss is not None):
                print(
                    f" stop_loss={self.stop_loss * 100:.3f}%, take_profit={self.take_profit * 100:.3f}%"
                )
            print(
                f" enter_at={self.enter_threshold:.3f}, close_at={self.close_threshold:.3f}"
            )

    def __call__(self, *args, **kwargs):
        while not self.done:
            # step must be start_step adjusted cause one can start and end backtest at any point in df
            _step = self.current_step - self.start_step
            if self.signals[_step] >= self.enter_threshold:
                if self.in_position:
                    action, self.signals[_step] = 0, 0
                else:
                    action = 1
                # action = 1
            elif self.signals[_step] <= -self.close_threshold:
                if not self.in_position:
                    action, self.signals[_step] = 0, 0
                else:
                    action = 2
                # action = 2
            else:
                action = 0
            self.step(action)
            if action == 2:
                i = _step
                self.weights[i] += self.percentage_profit
                while self.signals[i] != 1:
                    i -= 1
                    self.weights[i] += self.percentage_profit
            if self.visualize:
                # current_step manipulation just to synchronize plot rendering
                # could be fixed by calling .render() inside .step() just before return statement
                self.current_step -= 1
                self.render(indicator_or_reward=self.signals[_step])
                self.current_step += 1
        return None, self.reward, self.done, False, self.info

# class SignalExecuteFuturesEnv(FuturesBacktest):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         if 'position_ratio' in kwargs:
#             self.position_ratio = kwargs['position_ratio']/100 if kwargs['position_ratio']>1 else kwargs['position_ratio']
#         else:
#             self.position_ratio = 1.0
#         if 'save_ratio' in kwargs:
#             self.save_ratio = kwargs['save_ratio'] / 100 if kwargs['save_ratio']>1 else \
#                 kwargs['save_ratio']
#         else:
#             self.save_ratio = None
#         self.leverage = kwargs['leverage'] if 'leverage' in kwargs else 1
#         self.stop_loss = kwargs['stop_loss'] if 'stop_loss' in kwargs else None
#         self.take_profit = kwargs['take_profit'] if 'take_profit' in kwargs else None
#         self.long_enter_threshold = kwargs['long_enter_at'] if 'long_enter_at' in kwargs else 1.0
#         self.long_close_threshold = kwargs['long_close_at'] if 'long_close_at' in kwargs else 1.0
#         self.short_enter_threshold = kwargs['short_enter_at'] if 'short_enter_at' in kwargs else 1.0
#         self.short_close_threshold = kwargs['short_close_at'] if 'short_close_at' in kwargs else 1.0
#         self.signals = choice([-1, 0, 1], size=self.total_steps)
#
#     def reset(self, *args, **kwargs):
#         if 'position_ratio' in kwargs:
#             self.position_ratio = kwargs['position_ratio'] / 100 if kwargs['position_ratio'] > 1 else kwargs[
#                 'position_ratio']
#         else:
#             self.position_ratio = 1.0
#         if 'save_ratio' in kwargs:
#             self.save_ratio = kwargs['save_ratio'] / 100 if kwargs['save_ratio'] > 1 else \
#                 kwargs['save_ratio']
#         else:
#             self.save_ratio = None
#         self.leverage = kwargs['leverage'] if 'leverage' in kwargs else 1
#         self.stop_loss = kwargs['stop_loss'] if 'stop_loss' in kwargs else None
#         self.take_profit = kwargs['take_profit'] if 'take_profit' in kwargs else None
#         self.long_enter_threshold = kwargs['long_enter_at'] if 'long_enter_at' in kwargs else 1.0
#         self.long_close_threshold = kwargs['long_close_at'] if 'long_close_at' in kwargs else 1.0
#         self.short_enter_threshold = kwargs['short_enter_at'] if 'short_enter_at' in kwargs else 1.0
#         self.short_close_threshold = kwargs['short_close_at'] if 'short_close_at' in kwargs else 1.0
#         return super().reset(*args, **kwargs)
#
#     def _finish_episode(self):
#         super()._finish_episode()
#         if self.verbose:
#             print(f' position_ratio={self.position_ratio:.2f}, leverage={self.leverage}')
#             if (self.take_profit is not None) and (self.stop_loss is not None):
#                 print(f' stop_loss={self.stop_loss * 100:.3f}%, take_profit={self.take_profit * 100:.3f}%')
#             print(f' long_enter_at={self.long_enter_threshold:.3f}, long_close_at={self.long_close_threshold:.3f}')
#             print(f' short_enter_at={self.short_enter_threshold:.3f}, short_close_at={self.short_close_threshold:.3f}')
#
#     def __call__(self, *args, **kwargs):
#         while not self.done:
#             # step must be start_step adjusted cause one can start and end backtest at any point in df
#             _step = self.current_step - self.start_step
#             action = 0
#             if self.qty == 0:
#                 if self.signals[_step] >= self.long_enter_threshold:
#                     action = 1
#                 elif self.signals[_step] <= -self.short_enter_threshold:
#                     action = 2
#             elif self.qty > 0 and self.signals[_step] <= -self.long_close_threshold:
#                 action = 2
#             elif self.qty < 0 and self.signals[_step] >= self.short_close_threshold:
#                 action = 1
#             self.step(action)
#             if self.visualize:
#                 # current_step manipulation just to synchronize plot rendering
#                 # could be fixed by calling .render() inside .step() just before return statement
#                 self.current_step -= 1
#                 self.render(indicator_or_reward=self.signals[_step])
#                 self.current_step += 1
#         return None, self.reward, self.done, False, self.info
