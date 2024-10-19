from collections import deque
from math import copysign
from time import sleep
from warnings import warn

from gym import spaces
# from gymnasium import spaces
from numpy import array, inf, full, float32, zeros, hstack, isnan

from .base import SpotBacktest


class SpotTakerRL(SpotBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Observation space #
        # other_obs_count are observations like current PnL, account balance, asset quantity etc. #
        other_obs_count = 10
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

    def _get_report_header(self):
        base_header = super()._get_report_header()
        return base_header + ["obs_space_bal_to_init_ratio",
                              "obs_space_pos_to_bal_ratio",
                              "obs_space_in_position",
                              "obs_space_position_hold_c",
                              "obs_space_passive_c",
                              "obs_space_pnl",
                              "obs_space_sl_dist",
                              "obs_space_tp_dist",
                              "obs_space_profit_loss_hold_ratio"]

    def _get_report_row(self):
        base_row = super()._get_report_row()
        # Skip first element of trading_data as it's already in base_row
        base_row.extend(self.trade_data[1:])
        return base_row

    # Reset the state of the environment to an initial state
    def reset(self, **kwargs):
        first_obs = super().reset(**kwargs)
        self.trade_data = [
            self.qty,
            self.balance / self.init_balance,
            self.position_size / self.balance,
            self.in_position,
            self.in_position_counter,
            self.passive_counter,
            self.pnl,
            0,
            0,
            self.profit_hold_counter / self.loss_hold_counter if self.loss_hold_counter > 0 else 0
        ]
        if isnan(self.trade_data).any():
            raise ValueError(f"NaNs in trade_data {self.trade_data}")
        self.lookback = deque([zeros(self.obs_space_dims) for _ in range(self.lookback_size)],
                              maxlen=self.lookback_size)
        self.lookback.append(hstack((first_obs, self.trade_data)))
        # return hstack((first_obs, self.trade_data)), self.info
        return array(self.lookback)

    # Get the data points for the given current_step
    def _next_observation(self):
        df_features = super()._next_observation()
        close_price = self.df[self.current_step, 3]
        dist_to_sl = (
            (close_price - self.stop_loss_price)
            / close_price
            if self.stop_loss is not None and self.in_position
            else 0
        )
        dist_to_tp = (
            (self.take_profit_price - close_price)
            / close_price
            if self.take_profit is not None and self.in_position
            else 0
        )
        self.trade_data = [
            self.qty,
            self.balance / self.init_balance,
            self.position_size / (self.balance + (self.position_size + (self.position_size * self.pnl))),
            self.in_position,
            self.in_position_counter,
            self.passive_counter,
            self.pnl,
            dist_to_sl,
            dist_to_tp,
            self.profit_hold_counter / self.loss_hold_counter if self.loss_hold_counter > 0 else 0
        ]
        if isnan(self.trade_data).any():
            raise ValueError(f"NaNs in trade_data {self.trade_data}")
        self.lookback.append(hstack((df_features, self.trade_data)))
        # print(f"array(self.lookback) {array(self.lookback)}")
        return array(self.lookback)

    def _calculate_reward(self):

        # # Nagroda lub kara za zmianę wartości portfela
        # if self.in_position:
        #     # Nagroda proporcjonalna do niezrealizowanego zysku/straty
        #     self.reward = self.pnl
        # elif self.passive_penalty and (self.passive_counter > self.steps_passive_penalty):
        #     self.reward -= 0.1
        # elif self.position_closed:
        #     # Nagroda proporcjonalna do zrealizowanego zysku/straty
        #     self.reward = 10 * (self.absolute_profit / self.init_balance)
        # else:
        #     self.reward = 0
        #
        # # Ograniczenie wartości nagrody do zakresu [-1, 1]
        # self.reward = max(min(self.reward, 1), -1)
        #
        # return self.reward

        # prev, continiuous rew
        if self.in_position:
            self.cum_pnl += self.pnl
            self.reward = self.pnl
        elif self.position_closed:
            if self.pnl < 0 and self.cum_pnl < 0:
                self.reward = -1 * self.cum_pnl * copysign(1, self.pnl)
            else:
                self.reward = self.cum_pnl * copysign(1, self.pnl)
            self.position_closed, self.cum_pnl = 0, 0
        elif self.passive_penalty and (self.passive_counter > self.steps_passive_penalty):
            self.reward = -1
        else:
            self.reward = 0
        return self.reward

    def step(self, action):
        self.stop_loss = action[1]
        self.take_profit = action[2]
        obs, _, _, _, _ = super().step(round(action[0]))
        sleep(1)
        if isnan(obs).any():
            raise ValueError(f"NaNs in df_features {obs}")
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


# TODO: Create spot market maker environment
class SpotMakerRL(SpotBacktest):
    def __init__(self, df, *args, **kwargs):
        super().__init__(df, *args, **kwargs)
        raise NotImplemented(
            "Market-maker version of this environment is not implemented yet."
        )
