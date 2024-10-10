from warnings import warn

from gym import spaces

# from gymnasium import spaces
from numpy import array, inf, full, float32, zeros, hstack, isnan
from collections import deque

from .base import SpotBacktest

from time import sleep


class SpotTakerRL(SpotBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Observation space #
        # other_obs_count are observations like current PnL, account balance, asset quantity etc. #
        other_obs_count = 11
        self.lookback_size = kwargs['lookback_size']
        # As default, don't use ohlcv values from dataframe as features/obs space
        if self.exclude_cols_left < 5:
            warn(
                f"OHLCV values are not excluded from features/observation space (exclude_cols_left={self.exclude_cols_left})"
            )
        self.obs_space_dims = len(self.df[0, self.exclude_cols_left:]) + other_obs_count
        obs_lower_bounds = full((self.lookback_size, self.obs_space_dims), -inf)
        obs_upper_bounds = full((self.lookback_size, self.obs_space_dims), inf)
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds, dtype=float32)
        # self.action_space = spaces.Dict({'trade': spaces.Discrete(2),
        #                                  'stop_loss': spaces.Box(low=0, high=1, shape=()),
        #                                  'take_profit': spaces.Box(low=0, high=1, shape=())
        #                                  })
        self.action_space = spaces.Box(low=array([0.0, 0.001, 0.001]), high=array([2.0, 0.250, 0.250]),
                                       dtype=float32)

        self.lookback = deque([zeros(self.obs_space_dims).copy() for _ in range(self.lookback_size)],
                              maxlen=self.lookback_size)
        print(f"    observation_space {self.observation_space}")

    # Reset the state of the environment to an initial state
    def reset(self, **kwargs):
        first_obs = super().reset(**kwargs)
        trade_data = [
            self.qty,
            self.balance / self.init_balance,
            self.position_size / self.balance,
            self.in_position,
            self.in_position_counter,
            self.passive_counter,
            self.pnl,
            0,
            0,
            self.profit_hold_counter,
            self.loss_hold_counter,
        ]
        if isnan(trade_data).any():
            raise ValueError(f"NaNs in trade_data {trade_data}")
        self.lookback = deque([zeros(self.obs_space_dims) for _ in range(self.lookback_size)],
                              maxlen=self.lookback_size)
        self.lookback.append(hstack((first_obs, trade_data)))
        # return hstack((first_obs, trade_data)), self.info
        return array(self.lookback)

    # Get the data points for the given current_step
    def _next_observation(self):
        df_features = super()._next_observation()
        dist_to_sl = (
            (self.df[self.current_step, 3] - self.stop_loss_price)
            / self.df[self.current_step, 3]
            if self.stop_loss is not None
            else 0
        )
        dist_to_tp = (
            (self.df[self.current_step, 3] - self.take_profit_price)
            / self.df[self.current_step, 3]
            if self.take_profit is not None
            else 0
        )
        trade_data = [
            self.qty,
            self.balance / self.init_balance,
            self.position_size / self.balance if self.balance > 0 else 0,
            self.in_position,
            self.in_position_counter,
            self.passive_counter,
            self.pnl,
            dist_to_sl,
            dist_to_tp,
            self.profit_hold_counter,
            self.loss_hold_counter,
        ]
        if isnan(trade_data).any():
            raise ValueError(f"NaNs in trade_data {trade_data}")
        self.lookback.append(hstack((df_features, trade_data)))
        # print(f"array(self.lookback) {array(self.lookback)}")
        return array(self.lookback)

    def _calculate_reward(self):
        if self.passive_penalty and (self.passive_counter > self.steps_passive_penalty):
            self.reward = -1
            return self.reward
        if self.in_position:
            self.reward = self.df[self.current_step, 3] / self.enter_price - 1
        else:
            self.reward = 0
        return self.reward

        # if self.in_position:
        #     _pnl = self.df[self.current_step, 3] / self.enter_price - 1
        #     _balance = self.balance + (
        #             self.position_size + (self.position_size * _pnl)
        #     )
        # else:
        #     _balance = self.balance
        # self.reward = (_balance - self.init_balance)/self.init_balance
        # return self.reward


        # Position closed/sold #
        # if self.position_closed:
        #     last_pnl = self.PLs_and_ratios[-1][0]
        #     if last_pnl > 0:
        #         self.reward = (
        #                 10 * last_pnl * (
        #                     self.balance / self.init_balance)
        #         )
        #     elif last_pnl < 0:
        #         self.reward = (
        #                 10 * last_pnl * (
        #                     self.init_balance / self.balance)
        #         )
        #     self.position_closed = 0
        # # In Position #
        # elif self.in_position:
        #     # self.reward = self.pnl / self.in_position_counter if self.in_position_counter > 1 else 0
        #     self.reward = 0
        # else:
        #     self.reward = 0
        # if self.passive_penalty and (self.passive_counter > self.steps_passive_penalty):
        #     self.reward = -1
        # # print(f'reward: {self.reward}')
        # return self.reward

    def step(self, action):
        self.stop_loss = action[1]
        self.take_profit = action[2]
        obs, _, _, _, _ = super().step(round(action[0]))
        self._calculate_reward()
        # current_step manipulation just to synchronize plot rendering
        # could be fixed by calling .render() inside .step() just before return statement
        # print(f'action {action} reward {self.reward}')
        if self.visualize:
            self.current_step -= 1
            self.render()
            self.current_step += 1
        # sleep(5)
        # return obs, self._calculate_reward(), self.done, False, self.info
        return obs, self.reward, self.done, self.info

    def render(self, visualize=False, *args, **kwargs):
        super().render(
            *args, indicator_or_reward=self.reward, visualize=visualize, **kwargs
        )


class SpotMakerRL(SpotBacktest):
    def __init__(self, df, *args, **kwargs):
        super().__init__(df, *args, **kwargs)
        raise NotImplemented(
            "Market-maker version of this environment is not implemented yet."
        )
