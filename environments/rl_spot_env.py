from collections import deque
from warnings import warn

import numpy as np
from gym import spaces
# from gymnasium import spaces
from numpy import array, inf, full, float32, zeros, hstack

from .base import SpotBacktest


class DiscreteSpotTakerRL(SpotBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        trade_obs_space = 6  # ones like current pnl, current qty etc. (trade_data)
        self.lookback_size = kwargs["lookback_size"]
        # As default, don't use ohlcv values from dataframe as features/obs space
        if self.exclude_cols_left < 5:
            warn(
                f"OHLCV values are not excluded from features/observation space (exclude_cols_left={self.exclude_cols_left})"
            )
        self.obs_space_dims = (
                len(self.df[0, self.exclude_cols_left:]) + trade_obs_space
        )
        obs_lower_bounds = full((self.lookback_size, self.obs_space_dims), -inf)
        obs_upper_bounds = full((self.lookback_size, self.obs_space_dims), inf)
        self.observation_space = spaces.Box(
            low=obs_lower_bounds, high=obs_upper_bounds, dtype=float32
        )
        self.action_space = spaces.Discrete(3)
        self.lookback = deque(
            [zeros(self.obs_space_dims).copy() for _ in range(self.lookback_size)],
            maxlen=self.lookback_size,
        )
        print(f"    observation_space: {self.observation_space}")
        print(f"    action_space: {self.action_space}")

    def _get_report_header(self):
        base_header = super()._get_report_header()
        return base_header + [
            "reward",
            "fullbal_to_init_ratio",
            "in_position",
            "position_hold_adj_c",
            "passive_adj_c",
            "pnl",
        ]

    def _get_report_row(self):
        base_row = super()._get_report_row()
        # Skip first element of trading_data (qty) as it's already in base_row
        base_row.extend([self.reward] + self.trade_data[1:])
        return base_row

    def reset(self, **kwargs):
        first_obs = super().reset(**kwargs)
        _full_bal = self._get_full_balance()
        self.trade_data = [
            self.qty,
            (_full_bal / self.init_balance) - 1,
            self.in_position,
            (
                self.in_position_counter / self.max_steps
                if self.max_steps > 0
                else self.in_position_counter / self.df.shape[0]
            ),
            min(self.passive_counter / self.steps_passive_penalty, 1),
            self.pnl,
        ]
        self.lookback = deque(
            [zeros(self.obs_space_dims) for _ in range(self.lookback_size)],
            maxlen=self.lookback_size,
        )
        self.lookback.append(hstack((first_obs, self.trade_data)))
        # return hstack((first_obs, self.trade_data)), self.info
        return array(self.lookback)

    def _next_observation(self):
        df_features = super()._next_observation()
        _full_bal = self._get_full_balance()
        self.trade_data = [
            self.qty,
            (_full_bal / self.init_balance) - 1,
            self.in_position,
            (
                self.in_position_counter / self.max_steps
                if self.max_steps > 0
                else self.in_position_counter / self.df.shape[0]
            ),
            min(self.passive_counter / self.steps_passive_penalty, 1),
            self.pnl,
        ]
        # if isnan(self.trade_data).any():
        #     raise ValueError(f"NaNs in trade_data {self.trade_data}")
        self.lookback.append(hstack((df_features, self.trade_data)))
        return array(self.lookback)

    def _calculate_reward(self):
        if self.in_position:
            # self.reward = self.pnl/(self.in_position_counter+1)**2
            self.reward = self.pnl / self.in_position_counter
        elif self.position_closed:
            relative_profit = self.absolute_profit / self.init_balance
            current_close = self.df[self.current_step, 3]
            start = max(0, self.current_step - self.lookback_size)
            end = min(len(self.df), self.current_step + self.lookback_size)
            local_max = np.max(
                self.df[start:end, 1]
            )  # highest High price in local steps range
            local_min = np.min(
                self.df[start:end, 2]
            )  # lowest Low price in local steps range
            max_min_diff = local_max - local_min
            if self.absolute_profit > 0:
                peak_scaling = 1 - (abs(current_close - local_max) / max_min_diff)
                self.reward = relative_profit + relative_profit * min(peak_scaling, 1)
            elif self.absolute_profit < 0:
                peak_scaling = 1 - (abs(current_close - local_min) / max_min_diff)
                self.reward = relative_profit + relative_profit * min(peak_scaling, 1)
            self.position_closed, self.cum_pnl = 0, 0
        elif self.passive_penalty and (
                self.passive_counter > self.steps_passive_penalty
        ):
            self.reward -= 0.05
        else:
            self.reward = 0
        self.reward = max(min(self.reward, 1), -1)
        return self.reward

    def step(self, action):
        obs, _, _, _, _ = super().step(action)
        self._calculate_reward()

        # current_step manipulation just to synchronize plot rendering
        # could be fixed by calling .render() inside .step() just before return statement
        if self.visualize:
            self.current_step -= 1
            self.render()
            self.current_step += 1
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
