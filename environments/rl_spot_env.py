from collections import deque
from warnings import warn

from gymnasium import spaces
from numpy import array, inf, full, float32, zeros, hstack, mean, std

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
        obs_lower_bounds = full((self.lookback_size, self.obs_space_dims), -inf, dtype=float32)
        obs_upper_bounds = full((self.lookback_size, self.obs_space_dims), inf, dtype=float32)
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
        # Skip the first element of trading_data (qty) as it's already in base_row
        base_row.extend([self.reward] + self.trade_data[1:])
        return base_row

    def reset(self, **kwargs):
        first_obs, reset_info = super().reset(**kwargs)
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
        # return array(self.lookback) # gym
        return array(self.lookback, dtype=float32), reset_info  # gymnasium

    def _next_observation(self):
        # df_features = super()._next_observation()
        self.trade_data = [
            self.qty,
            (self._get_full_balance() / self.init_balance) - 1,
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
        self.lookback.append(hstack((super()._next_observation(), self.trade_data)))
        return array(self.lookback)

    # def _calculate_reward(self):
    #     if self.in_position:
    #         # self.reward = self.pnl/(self.in_position_counter+1)**2
    #         self.reward = self.pnl / self.in_position_counter
    #     elif self.position_closed:
    #         relative_profit = self.absolute_profit / self.init_balance
    #         current_close = self.df[self.current_step, 3]
    #         start = max(0, self.current_step - self.lookback_size)
    #         end = min(len(self.df), self.current_step + self.lookback_size)
    #         local_max = np.max(
    #             self.df[start:end, 1]
    #         )  # highest High price in local steps range
    #         local_min = np.min(
    #             self.df[start:end, 2]
    #         )  # lowest Low price in local steps range
    #         max_min_diff = local_max - local_min
    #         if self.absolute_profit > 0:
    #             peak_scaling = 1 - (abs(current_close - local_max) / max_min_diff)
    #             self.reward = relative_profit + relative_profit * min(peak_scaling, 1)
    #         elif self.absolute_profit < 0:
    #             peak_scaling = 1 - (abs(current_close - local_min) / max_min_diff)
    #             self.reward = relative_profit + relative_profit * min(peak_scaling, 1)
    #         self.position_closed, self.cum_pnl = 0, 0
    #     elif self.passive_penalty and (
    #             self.passive_counter > self.steps_passive_penalty
    #     ):
    #         self.reward -= 0.05
    #     else:
    #         self.reward = 0
    #     self.reward = max(min(self.reward, 1), -1)
    #     return self.reward

    # Simplest version
    def _calculate_reward(self):
        if self.position_closed:
            # print(f'(self.percentage_profit+1) {(self.percentage_profit+1)}')
            # relative_profit = self.absolute_profit / self.init_balance
            # bonus = min(0.5, relative_profit)
            # gain_bonus = ((self.balance/self.init_balance)-1)/10
            if self.absolute_profit > 0:
                self.reward = self.percentage_profit * 10
            elif self.absolute_profit < 0:
                self.reward = self.percentage_profit * 10
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

        self.PNL_arrays = array(self.PLs_and_ratios)
        if self.PNL_arrays.shape[0] > 1:
            mean_pnl = mean(self.PNL_arrays[:, 0])
            std_pnl = std(self.PNL_arrays[:, 0])
            profits = self.PNL_arrays[:, 0][self.PNL_arrays[:, 0] > 0]
            losses = self.PNL_arrays[:, 0][self.PNL_arrays[:, 0] < 0]
            profit_mean = mean(profits) if profits.size > 0 else 0.0
            loss_mean = mean(losses) if losses.size > 0 else 0.0
            profit_std = std(profits) if profits.size > 0 else 0.0
            loss_std = std(losses) if losses.size > 0 else 0.0
            win_rate = mean(self.PNL_arrays[:, 1])
            profit_loss_ratio = (
                abs(profit_mean / loss_mean)
                if (profit_mean * loss_mean) != 0 else 1.0
            )
            steps = self.max_steps if self.max_steps > 0 else self.total_steps
            in_gain_ratio = self.with_gain_c / max(
                steps - self.profit_hold_counter - self.loss_hold_counter - self.episode_orders,
                1
            )
        else:
            mean_pnl = 0.0
            std_pnl = 0.0
            profit_mean = 0.0
            loss_mean = 0.0
            profit_std = 0.0
            loss_std = 0.0
            win_rate = 0.0
            profit_loss_ratio = 0.0
            in_gain_ratio = 0.0

        cur_bal = self._get_full_balance()
        self.info.update({
            "balance": cur_bal,
            "return": 100 * (cur_bal / self.init_balance - 1),
            # "max_balance": self.max_balance,
            # "min_balance": self.min_balance,
            "pnl": self.pnl,
            "step_reward": self.reward,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "mean_pnl": mean_pnl,
            "std_pnl": std_pnl,
            "mean_profit": profit_mean,
            "mean_loss": loss_mean,
            "profit_std": profit_std,
            "loss_std": loss_std,
            "in_gain_ratio": in_gain_ratio,
        })

        # self.info.update({
        #                 "balance": self._get_full_balance(),
        #                 "pnl": self.pnl,
        #                 "step_reward": self.reward
        #                                 })
        # return obs, self.reward, self.done, self.info # gym
        return obs, self.reward, self.done, False, self.info  # gymnasium

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
