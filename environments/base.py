import logging
# from random import randint
import random as py_random
from csv import writer
from datetime import datetime as dt
from math import copysign, floor, sqrt
from sys import stdout
from time import time

from gymnasium import spaces, Env
from gymnasium.utils import seeding
from matplotlib.dates import date2num
from numpy import array, mean, std, searchsorted, float32, ascontiguousarray, random
from pandas import to_datetime

from definitions import REPORT_DIR
from utils.visualize import TradingGraph


class SpotBacktest(Env):
    def __init__(
            self,
            df,
            start_date="",
            end_date="",
            max_steps=0,
            exclude_cols_left=1,
            no_action_finish=2_880,
            steps_passive_penalty=0,
            init_balance=1_000,
            position_ratio=1.0,
            save_ratio=None,
            stop_loss=None,
            take_profit=None,
            fee=0.0002,
            coin_step=0.001,
            slippage=None,
            slipp_std=0,
            visualize=False,
            render_range=120,
            verbose=True,
            report_to_file=False,
            seed=37,
            *args,
            **kwargs,
    ):
        # Configure logging for this instance
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(stdout)],
        )
        self.logger = logging.getLogger(__name__)

        self.creation_t = time()
        self.df_input = df
        self.start_date = start_date
        self.end_date = end_date
        self.max_steps = max_steps
        self.exclude_cols_left = exclude_cols_left
        self.no_action_finish = no_action_finish
        self.steps_passive_penalty = steps_passive_penalty
        self.init_balance = init_balance
        self.position_ratio = position_ratio
        self.save_ratio = save_ratio
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.fee = fee
        self.coin_step = coin_step
        self.slippage = slippage
        self.slipp_std = slipp_std
        self.visualize = visualize
        self.render_range = render_range
        self.verbose = verbose
        self.report_to_file = report_to_file
        self.args = args
        self.kwargs = kwargs

        self._process_input_data()
        self._initialize_parameters()
        self._setup_visualization()
        self._setup_reporting()
        self.seed(seed)

    def seed(self, seed):
        self.gym_np_random, seed = seeding.np_random(seed)
        py_random.seed(seed)
        random.seed(seed)
        return [seed]

    def _process_input_data(self):
        if self.df_input.isnull().values.any():
            nan_counts = self.df_input.isnull().sum()
            nan_columns = nan_counts[nan_counts > 0].index.tolist()
            raise ValueError(
                f"Input dataframe contains NaN values in columns: {nan_columns}"
            )

        self.dates = to_datetime(self.df_input["Opened"])
        self.df = ascontiguousarray(
            self.df_input.drop(columns=["Opened"]).to_numpy(dtype=float32)
        )
        self.df_features = self.df_input.columns.tolist()

        if self.start_date != "" and self.end_date != "":
            start_date_dt = to_datetime(self.start_date)
            end_date_dt = to_datetime(self.end_date)
            self.logger.debug(
                f"Types of start_date {type(start_date_dt)} end_date {type(end_date_dt)}"
            )
            self.logger.debug(f"Dates dtype {self.dates.dtype}")
            self.start_index = searchsorted(self.dates, start_date_dt, side="left")
            self.end_index = searchsorted(self.dates, end_date_dt, side="right") - 1
        else:
            self.start_index = 0
            self.end_index = self.dates.shape[0] - 1

        trade_range_size = self.df[self.start_index: self.end_index, :].shape[0]
        self.trade_range_size = trade_range_size
        if trade_range_size < self.max_steps:
            raise ValueError(
                "max_steps is larger than the number of rows in the dataframe"
            )

    def _initialize_parameters(self):
        if self.verbose:
            self.logger.info(f"Environment ({self.__class__.__name__}) created.")
            self.logger.info(f"Fee: {self.fee}, coin_step: {self.coin_step}")
            self.logger.info(
                f"full_df_size: {self.df.shape[0]}, trade_range_size: {self.trade_range_size}, "
                f"max_steps: {self.max_steps} ({self.max_steps / self.df.shape[0] * 100:.2f}%)"
            )
            self.logger.info(f"no_action_finish: {self.no_action_finish}")
            self.logger.info(f"Sample df row: {self.df[-1, self.exclude_cols_left:]}")
            self.logger.info(f"Slippage statistics (avg, stddev): {self.slippage}")
            self.logger.info(
                f"init_balance: {self.init_balance}, position_ratio: {self.position_ratio}"
            )
            self.logger.info(
                f"save_ratio: {self.save_ratio}, stop_loss: {self.stop_loss}, take_profit: {self.take_profit}"
            )

        if self.slippage is not None:
            self.buy_factor = (
                    self.slippage["buy"][0] + self.slippage["buy"][1] * self.slipp_std
            )
            self.sell_factor = (
                    self.slippage["sell"][0] - self.slippage["sell"][1] * self.slipp_std
            )
            self.stop_loss_factor = (
                    self.slippage["stop_loss"][0]
                    - self.slippage["stop_loss"][1] * self.slipp_std
            )
            self.take_profit_factor = 1.0  # To be updated if needed
        else:
            self.buy_factor = 1.0
            self.sell_factor = 1.0
            self.stop_loss_factor = 1.0
            self.take_profit_factor = 1.0

        self.save_balance = 0.0
        self.cum_pnl = 0.0
        self.total_steps = len(self.df)
        if self.steps_passive_penalty == "auto":
            base = self.max_steps if self.max_steps > 0 else self.total_steps
            self.steps_passive_penalty = max(int(sqrt(base)), 1)
            self.passive_penalty = True
        elif self.steps_passive_penalty > 0:
            self.passive_penalty = True
        else:
            self.passive_penalty = False

        self.position_size = self.init_balance * self.position_ratio
        self.balance = self.init_balance

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)
        # TODO: Define observation_space appropriately

    def _setup_visualization(self):
        self.dates = date2num(self.dates)
        if self.visualize:
            self.time_step = self.dates[1] - self.dates[0]
            if self.verbose:
                self.logger.info(
                    f"Visualization enabled, time step: {self.time_step} (as a fraction of a day)"
                )
            # Initialize visualization (assuming TradingGraph is defined)
            self.visualization = TradingGraph(self.render_range, self.time_step)
        else:
            self.visualize = False
            if self.verbose:
                self.logger.info("Visualization is disabled or no date data provided.")

    def _setup_reporting(self):
        if self.report_to_file:
            self.filename = f'{REPORT_DIR}/envs/EP_{self.__class__.__name__}_{dt.now().strftime("%Y-%m-%d_%H-%M")}.csv'
            self.report_file = open(self.filename, "w", newline="")
            self.report_writer = writer(self.report_file)
            self.report_writer.writerow(self._get_report_header())
            if self.verbose:
                self.logger.info(f"Environment will report to file: {self.filename}")
                self.logger.debug(f"File header: {self._get_report_header()}")

    def _get_report_header(self):
        return self.df_features + [
            "trade_no",
            "leverage",
            "trade_side",
            "position_size",
            "quantity",
            "balance",
            "save_balance",
            "profit_to_pos_ratio",
            "cumulative_fee",
        ]

    def _get_report_row(self):
        return [
            self.dates[self.current_step],
            *self.df[self.current_step, :],
            self.episode_orders,
            1,  # Constant leverage for spot env
            self.last_order_type,
            self.position_size,
            self.qty,
            self.balance,
            self.save_balance,
            self.absolute_profit / self.position_size,
            self.cumulative_fees,
        ]

    def _report_to_file(self):
        self.report_writer.writerow(self._get_report_row())

    def reset(self, **kwargs):
        self.creation_t = time()
        self.done = False
        self.reward = 0
        self.cum_pnl = 0.0
        if self.visualize:
            self.visualization = TradingGraph(self.render_range, self.time_step)
        if self.report_to_file:
            self.filename = f'{REPORT_DIR}/envs/EP_{self.__class__.__name__}_{str(dt.today()).replace(":", "-")[:-3]}.csv'
            self.report_file = open(self.filename, "w", newline="")
            self.report_writer = writer(self.report_file)
            self.report_writer.writerow(self._get_report_header())
        self.last_order_type = ""
        self.info = {}
        self.PLs_and_ratios = []
        self.balance = self.init_balance
        self.position_size = self.init_balance * self.position_ratio
        self.prev_bal = 0
        self.enter_price = 0
        self.stop_loss_price, self.take_profit_price = 0, 0
        self.qty = 0
        self.pnl = 0
        self.percentage_profit, self.absolute_profit = 0.0, 0.0
        self.SL_losses, self.cumulative_fees, self.liquidations, self.take_profits_c = (
            0,
            0,
            0,
            0,
        )
        (
            self.in_position,
            self.in_position_counter,
            self.position_closed,
            self.passive_counter,
        ) = (0, 0, 0, 0)
        self.episode_orders, self.with_gain_c = 0, 1
        self.good_trades_count, self.bad_trades_count = 1, 1
        self.max_drawdown, self.max_profit = 0, 0
        self.loss_hold_counter, self.profit_hold_counter = 0, 0
        self.max_balance = self.min_balance = self.balance
        self.save_balance = 0.0
        if self.max_steps > 0:
            # self.start_step = randint(self.start_index, self.end_index - self.max_steps) # older version
            self.start_step = self.gym_np_random.integers(self.start_index, self.end_index - self.max_steps + 1)
            self.end_step = self.start_step + self.max_steps - 1
        else:
            self.start_step = self.start_index
            self.end_step = self.end_index
        self.current_step = self.start_step
        self.obs = iter(
            self.df[self.start_step: self.end_step, self.exclude_cols_left:]
        )
        # return self.df[self.current_step, self.exclude_cols_left:]
        # return next(self.obs) # gym
        return next(self.obs), {}  # gymnasium

    def _get_full_balance(self, include_position=False):
        if self.in_position:
            if include_position:
                _pnl = self.df[self.current_step, 3] / self.enter_price - 1
                return (
                        self.balance
                        + (self.position_size + (self.position_size * _pnl))
                        + self.save_balance
                )
            else:
                return self.prev_bal
        else:
            return self.balance + self.save_balance

    def _buy(self, price):
        if self.stop_loss is not None:
            self.stop_loss_price = round((1 - self.stop_loss) * price, 2)
        if self.take_profit is not None:
            self.take_profit_price = round((1 + self.take_profit) * price, 2)
        # Considering random factor as in real world scenario #
        # price = self._random_factor(price, 'market_buy')
        adj_price = price * self.buy_factor
        # When there is no fee, subtract 1 just to be sure balance can buy this amount #
        step_adj_qty = floor(self.position_size / (adj_price * self.coin_step))
        if step_adj_qty == 0:
            self._finish_episode()
            return
        self.last_order_type = "open_long"
        self.in_position = 1
        self.in_position_counter = 1
        self.passive_counter = 0
        self.episode_orders += 1
        self.enter_price = adj_price
        self.qty = step_adj_qty * self.coin_step
        self.position_size = self.qty * adj_price
        self.prev_bal = self.balance
        self.balance -= self.position_size
        fee = self.position_size * self.fee
        self.position_size -= fee
        self.cumulative_fees += fee
        self.absolute_profit = -fee
        # print(f'BOUGHT {self.qty} at {price}({adj_price})')

    def _sell(self, price, sl=False, tp=False):
        if sl:
            # price = self._random_factor(price, 'SL')
            # while price>self.enter_price:
            #     price = self._random_factor(price, 'SL')
            adj_price = price * self.stop_loss_factor
            if adj_price > self.enter_price:
                raise RuntimeError(
                    f"Stop loss price is above position enter price. (sl_factor={self.stop_loss_factor})"
                )
            self.last_order_type = "stop_loss_long"
        elif tp:
            self.take_profits_c += 1
            adj_price = price * self.take_profit_factor
            # TODO: add new order type for visualizations
            self.last_order_type = "take_profit_long"
        else:
            # price = self._random_factor(price, 'market_sell')
            adj_price = price * self.sell_factor
            self.last_order_type = "close_long"
        _value = self.qty * adj_price
        self.balance += round(_value, 2)
        fee = _value * self.fee
        self.balance -= fee
        self.cumulative_fees += fee
        self.percentage_profit = (self.balance / self.prev_bal) - 1
        self.absolute_profit = self.balance - self.prev_bal
        # print(f'SOLD {self.qty} at {price}({adj_price}) profit ${self.balance-self.prev_bal:.2f}')
        # PROFIT #
        if self.percentage_profit > 0:
            if self.save_ratio is not None:
                save_amount = self.absolute_profit * self.save_ratio
                self.save_balance += save_amount
                self.balance -= save_amount
            if self.balance >= self.max_balance:
                self.max_balance = self.balance
            self.good_trades_count += 1
            if self.max_profit == 0 or self.percentage_profit > self.max_profit:
                self.max_profit = self.percentage_profit
        # LOSS #
        elif self.percentage_profit < 0:
            if self.balance <= self.min_balance:
                self.min_balance = self.balance
            self.bad_trades_count += 1
            if self.max_drawdown == 0 or self.percentage_profit < self.max_drawdown:
                self.max_drawdown = self.percentage_profit
            if sl:
                self.SL_losses += self.absolute_profit
        self.PLs_and_ratios.append(
            (self.percentage_profit, self.good_trades_count / self.bad_trades_count)
        )
        self.position_size = self.balance * self.position_ratio
        # If balance minus position_size and fee is less or eq 0 #
        if self.position_size < (price * self.coin_step):
            self._finish_episode()
        self.qty = 0
        self.in_position = 0
        self.in_position_counter = 0
        self.passive_counter = 0
        self.position_closed = 1
        self.stop_loss_price = 0

    def _next_observation(self):
        try:
            return next(self.obs)
        except StopIteration:
            self.current_step -= 1
            self._finish_episode()
            return self.df[self.current_step, self.exclude_cols_left:]

    def step(self, action):
        self.last_order_type = ""
        self.absolute_profit = 0.0
        self.position_closed = 0
        self.sl_trigger, self.tp_trigger = 0, 0
        self.passive_counter += 1
        if self.in_position:
            high, low, close = self.df[self.current_step, 1:4]
            # print(f'low: {low}, close: {close}, self.enter_price: {self.enter_price}')
            self.in_position_counter += 1
            self.pnl = (close / self.enter_price) - 1
            if self.pnl >= 0:
                self.profit_hold_counter += 1
            else:
                self.loss_hold_counter += 1
            # Handling stop losses and take profits
            if (self.stop_loss is not None) and (low <= self.stop_loss_price):
                self.sl_trigger = 1
                self._sell(self.stop_loss_price, sl=True)
            elif (self.take_profit is not None) and (high >= self.take_profit_price):
                self.tp_trigger = 1
                self._sell(self.take_profit_price, tp=True)
            elif action == 2 and self.qty > 0:
                self._sell(close)
        elif action == 1:
            close = self.df[self.current_step, 3]
            self._buy(close)
            self.pnl = (close / self.enter_price) - 1
        elif (not self.episode_orders) and (
                (self.current_step - self.start_step) > self.no_action_finish
        ):
            self._finish_episode()
        else:
            # self.passive_counter += 1 # no penalty when in position
            if self.init_balance < self.balance + self.save_balance:
                self.with_gain_c += 1
        # Older version:
        # return self._next_observation(), self.reward, self.done, self.info
        # For now terminated == truncated (==self.done)
        self.current_step += 1
        if self.report_to_file:
            self._report_to_file()
        return self._next_observation(), self.reward, self.done, False, self.info

    def _finish_episode(self):
        # 1. House-keeping ― close report file, close any open position
        if self.report_to_file:
            self.report_file.close()
        if self.in_position:
            self._sell(self.enter_price)

        self.done = True

        # 2. Financial metrics
        self.PNL_arrays = array(self.PLs_and_ratios)  # (pnl, ratio)
        self.balance += self.save_balance
        gain = self.balance - self.init_balance
        total_return = (self.balance / self.init_balance) - 1
        benchmark_ret = (self.df[self.end_step, 3] /
                         self.df[self.start_step, 3]) - 1  # BTC buy-&-hold
        above_free = total_return - benchmark_ret

        hold_ratio = (
            self.profit_hold_counter / self.loss_hold_counter
            if (self.loss_hold_counter > 1 and self.profit_hold_counter > 1)
            else 1.0
        )

        # ――― PnL statistics ―――
        if len(self.PNL_arrays) > 1:
            mean_pnl, stddev_pnl = mean(self.PNL_arrays[:, 0]), std(self.PNL_arrays[:, 0])

            profits = self.PNL_arrays[:, 0][self.PNL_arrays[:, 0] > 0]
            losses = self.PNL_arrays[:, 0][self.PNL_arrays[:, 0] < 0]
            profits_mean = mean(profits) if profits.size > 0 else 0.0
            losses_mean = mean(losses) if losses.size > 0 else 0.0
            losses_stddev = std(losses) if losses.size > 0 else 0.0

            PnL_trades_ratio = mean(self.PNL_arrays[:, 1])
            PnL_means_ratio = (
                abs(profits_mean / losses_mean) if profits_mean * losses_mean != 0 else 1.0
            )
            slope_indicator = 1.000  # placeholder

            steps = self.max_steps if self.max_steps > 0 else self.total_steps
            in_gain_indicator = self.with_gain_c / max(
                steps - self.profit_hold_counter - self.loss_hold_counter - self.episode_orders,
                1,
            )
        else:
            mean_pnl = stddev_pnl = profits_mean = losses_mean = losses_stddev = 0.0
            PnL_trades_ratio = PnL_means_ratio = in_gain_indicator = 0.0
            slope_indicator = 0.000

        # 3. Reward shaping ― spread the final bonus along the episode
        episode_bonus = above_free
        step_count = max(self.end_step - self.start_step, 1)
        per_step_bonus = episode_bonus / step_count
        self.reward = per_step_bonus  # value returned on the last .step()
        self.terminal_bonus = episode_bonus  # fetched in step() when done==True

        # 4. Additional ratios
        sharpe_ratio = ((mean_pnl - benchmark_ret) / stddev_pnl) if stddev_pnl else -1
        sortino_ratio = ((total_return - benchmark_ret) / losses_stddev) if losses_stddev else -1
        exec_time = time() - self.creation_t

        # 5. Console logging
        if self.verbose:
            print(
                f"Episode finished: gain ${gain:.2f} ({total_return * 100:.2f}%), "
                f"gain/step ${gain / step_count:.5f}, fees ${self.cumulative_fees:.2f}, "
                f"SL_losses ${self.SL_losses:.2f}, take_profits {self.take_profits_c}\n"
                f" stop_loss {self.stop_loss}, take_profit {self.take_profit}, "
                f"save_ratio {self.save_ratio}, saved_balance ${self.save_balance:.2f}\n"
                f" trades {self.episode_orders:_}, good/bad "
                f"{self.good_trades_count - 1:_}/{self.bad_trades_count - 1:_}, "
                f"avg(p/l) {profits_mean * 100:.2f}%/{losses_mean * 100:.2f}%\n"
                f" PnL_trades_ratio {PnL_trades_ratio:.3f}, PnL_means_ratio {PnL_means_ratio:.3f}, "
                f"hold_ratio {hold_ratio:.3f}, PNL_mean {mean_pnl * 100:.2f}%\n"
                f" slope {slope_indicator:.4f}, in_gain {in_gain_indicator:.3f}, "
                f"sharpe {sharpe_ratio:.2f}, sortino {sortino_ratio:.2f}, "
                f"benchmark {benchmark_ret * 100:.2f}%, above_free {above_free * 100:.2f}%\n"
                f" per_step_reward {per_step_bonus:.6f}, exec_time {exec_time:.2f}s"
            )

        # 6. Info dict for callbacks/monitors
        self.info["episode"] = {"r": gain, "l": step_count}
        self.info = {
            "gain": gain,
            "above_benchmark": above_free,
            "PnL_means_ratio": PnL_means_ratio,
            "PnL_trades_ratio": PnL_trades_ratio,
            "hold_ratio": hold_ratio,
            "PNL_mean": mean_pnl,
            "slope_indicator": slope_indicator,
            "exec_time": exec_time,
        }

    def render(self, indicator_or_reward=None, visualize=False, *args, **kwargs):
        if visualize or self.visualize:
            if indicator_or_reward is None:
                indicator_or_reward = self.df[self.current_step, -1]
            render_row = [
                self.dates[self.current_step],
                *self.df[self.current_step, 0:4],
                indicator_or_reward,
                self._get_full_balance(),
            ]
            trade_info = [self.last_order_type, str(round(self.absolute_profit, 2))]
            #   print(f'trade_info {trade_info}')
            self.visualization.append(render_row, trade_info)
            self.visualization.render()


# TODO: Add report to file feature to Futures envs
class FuturesBacktest(SpotBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # https://www.binance.com/en/futures/trading-rules/perpetual/leverage-margin
        self.POSITION_TIER = {
            1: (125, 0.0040, 0),
            2: (100, 0.005, 50),
            3: (50, 0.01, 2_550),
            4: (20, 0.025, 122_550),
            5: (10, 0.05, 1_372_550),
            6: (5, 0.10, 5_372_550),
            7: (4, 0.125, 7_872_550),
            8: (3, 0.15, 10_872_550),
            9: (2, 0.25, 30_872_550),
            10: (1, 0.50, 105_872_550),
        }
        if "leverage" in kwargs:
            self.leverage = kwargs["leverage"]
        else:
            self.leverage = 1
        self.df_mark = kwargs["df_mark"].to_numpy()
        # BTCUSDTperp last 1Y mean=6.09e-05 stdev=6.52e-05, mean+2*stedv covers ~95,4% of variance
        # self.funding_rate = 0.01912 * (1/100)
        self.funding_rate = 0.01 * (1 / 100)
        print(
            f" df_mark_size: {len(self.df_mark)}, max_steps: {self.max_steps}({self.max_steps / len(self.df) * 100:.2f}%)"
        )

    def reset(self, **kwargs):
        self.margin = 0
        self.liquidation_price = 0
        self.tier = 0
        # self.stop_loss /= self.leverage
        self.liquidations = 0
        self.liquidation_losses = 0
        return super().reset(**kwargs)

    def _check_tier(self):
        # print('_check_tier')
        if self.position_size < 50_000:
            self.tier = 1
        elif 50_000 < self.position_size < 500_000:
            self.tier = 2
        elif 500_000 < self.position_size < 8_000_000:
            self.tier = 3
        elif 8_000_000 < self.position_size < 50_000_000:
            self.tier = 4
        elif 50_000_000 < self.position_size < 80_000_000:
            self.tier = 5
        elif 80_000_000 < self.position_size < 100_000_000:
            self.tier = 6
        elif 100_000_000 < self.position_size < 120_000_000:
            self.tier = 7
        elif 120_000_000 < self.position_size < 200_000_000:
            self.tier = 8
        elif 200_000_000 < self.position_size < 300_000_000:
            self.tier = 9
        elif 300_000_000 < self.position_size < 500_000_000:
            self.tier = 10
        if self.leverage > self.POSITION_TIER[self.tier][0]:
            # print(f' Leverage exceeds tier {self.tier} max', end=' ')
            # print(f'changing from {self.leverage} to {self.POSITION_TIER[self.tier][0]} (Balance:${self.balance}:.2f)')
            self.leverage = self.POSITION_TIER[self.tier][0]

    # def _check_margin(self):
    #   #print('_check_margin')
    #   if self.qty>0:
    #     min_price = self.df[self.current_step, 2]
    #   elif self.qty<0:
    #     min_price = self.df[self.current_step, 1]
    #   else:
    #     pass
    #     #print('co Ty tu robisz?')
    #   position_value = abs(self.qty*min_price)
    #   unrealized_PNL = abs(self.qty*self.enter_price/self.leverage)*self._get_pnl(min_price)
    #   # 1.25% Liquidation Clearance
    #   margin_balance = self.margin + unrealized_PNL - (position_value*0.0125) - (position_value*self.fee)
    #   maintenance_margin = position_value*self.POSITION_TIER[self.tier][1]-self.POSITION_TIER[self.tier][2]
    #   print(f'min_price:{min_price:.2f} position_value:{position_value:.2f} unrealized_PNL:{unrealized_PNL:.2f} Clearance:{(position_value*0.0125)} fee:{(position_value*self.fee)} margin:{self.margin} margin_balance:{margin_balance:.2f} maintenance_margin:{maintenance_margin:.2f} margin_ratio:{maintenance_margin/margin_balance*100}')
    #   if maintenance_margin>margin_balance:
    #     return True
    #   else:
    #     return False
    def _check_margin(self):
        # If in long position and mark Low below liquidation price
        if self.qty > 0:
            return self.liquidation_price >= self.df_mark[self.current_step, 3]
        # If in short position and mark High above liquidation price
        elif self.qty < 0:
            return self.liquidation_price <= self.df_mark[self.current_step, 2]
        return False

    def _get_pnl(self, price, update=False):
        if update and self.in_position:
            self.pnl = ((price / self.enter_price) - 1) * self.sign_leverage
            self.loss_hold_counter += self.pnl < 0
            self.profit_hold_counter += self.pnl > 0
            return self.pnl
        elif not self.in_position:
            return 0
        return ((price / self.enter_price) - 1) * self.sign_leverage

    def _open_position(self, side, price):
        if side == "long":
            adj_price = price * self.buy_factor
            if self.stop_loss is not None:
                self.stop_loss_price = round((1 - self.stop_loss) * price, 2)
            if self.take_profit is not None:
                self.take_profit_price = round((1 + self.take_profit) * price, 2)
            self.last_order_type = "open_long"
        elif side == "short":
            adj_price = price * self.sell_factor
            if self.stop_loss is not None:
                self.stop_loss_price = round((1 + self.stop_loss) * price, 2)
            if self.take_profit is not None:
                self.take_profit_price = round((1 - self.take_profit) * price, 2)
            self.last_order_type = "open_short"
        else:
            raise RuntimeError('side should be "long" or "short"')
        self._check_tier()
        adj_qty = floor(
            self.position_size * self.leverage / (adj_price * self.coin_step)
        )
        if adj_qty == 0:
            adj_qty = 1
            # print('Forcing adj_qty to 1. Calculated quantity possible to buy with given postion_size and coin_step equals 0')
        self.margin = (adj_qty * self.coin_step * adj_price) / self.leverage
        if self.margin > self.balance:
            self._finish_episode()
            return
        self.prev_bal = self.balance
        self.balance -= self.margin
        fee = self.margin * self.fee * self.leverage
        self.margin -= fee
        self.cumulative_fees -= fee
        self.absolute_profit = -fee
        self.in_position = 1
        self.episode_orders += 1
        self.enter_price = price
        if side == "long":
            self.qty = adj_qty * self.coin_step
        elif side == "short":
            self.qty = -1 * adj_qty * self.coin_step
        # for speeding up _get_pnl() method
        self.sign_leverage = copysign(1, self.qty) * self.leverage
        # https://www.binance.com/en/support/faq/how-to-calculate-liquidation-price-of-usd%E2%93%A2-m-futures-contracts-b3c689c1f50a44cabb3a84e663b81d93
        # 1,25% liquidation clearance fee https://www.binance.com/en/futures/trading-rules/perpetual/
        self.liquidation_price = (
                                         self.margin * (1 - 0.0125) - self.qty * self.enter_price
                                 ) / (abs(self.qty) * self.POSITION_TIER[self.tier][1] - self.qty)
        # print(f'OPENED {side} price:{price} adj_price:{adj_price} qty:{self.qty} margin:{self.margin} fee:{fee}')
        # sleep(10)

    def _close_position(self, price, liquidated=False, sl=False, tp=False):
        if sl:
            adj_price = price * self.stop_loss_factor
            if self.qty > 0:
                self.last_order_type = "stop_loss_long"
            elif self.qty < 0:
                self.last_order_type = "stop_loss_short"
        if tp:
            self.take_profits_c += 1
            adj_price = price * self.take_profit_factor
            # TODO: add new order type for visualizations
            if self.qty > 0:
                self.last_order_type = "take_profit_long"
            elif self.qty < 0:
                self.last_order_type = "take_profit_short"
        else:
            if self.qty > 0:
                adj_price = price * self.sell_factor
                self.last_order_type = "close_long"
            elif self.qty < 0:
                adj_price = price * self.buy_factor
                self.last_order_type = "close_short"
            else:
                raise RuntimeError("Bad call to _close_position, qty is 0")
        _position_value = abs(self.qty) * adj_price
        _fee = _position_value * self.fee
        if liquidated:
            if self.qty > 0:
                self.last_order_type = "liquidate_long"
            elif self.qty < 0:
                self.last_order_type = "liquidate_short"
            margin_balance = 0
            self.liquidation_losses -= self.margin
        else:
            unrealized_PNL = (
                                     abs(self.qty) * self.enter_price / self.leverage
                             ) * self._get_pnl(adj_price)
            margin_balance = self.margin + unrealized_PNL - _fee
        self.cumulative_fees -= _fee
        self.balance += margin_balance
        self.margin = 0
        percentage_profit = (self.balance / self.prev_bal) - 1
        self.absolute_profit = self.balance - self.prev_bal
        ### PROFIT
        if percentage_profit > 0:
            if self.save_ratio is not None:
                save_amount = self.absolute_profit * self.save_ratio
                self.save_balance += save_amount
                self.balance -= save_amount
            self.good_trades_count += 1
            if self.balance >= self.max_balance:
                self.max_balance = self.balance
            if (self.max_profit == 0) or (percentage_profit > self.max_profit):
                self.max_profit = percentage_profit
        ### LOSS
        elif percentage_profit < 0:
            self.bad_trades_count += 1
            if self.balance <= self.min_balance:
                self.min_balance = self.balance
            if (self.max_drawdown == 0) or (percentage_profit < self.max_drawdown):
                self.max_drawdown = percentage_profit
            if sl:
                self.SL_losses += self.absolute_profit
        self.PLs_and_ratios.append(
            (percentage_profit, self.good_trades_count / self.bad_trades_count)
        )
        self.position_size = self.balance * self.position_ratio
        self.qty = 0
        self.in_position = 0
        self.in_position_counter = 0
        self.position_closed = 1
        self.stop_loss_price = 0
        self.pnl = 0
        # print(f'CLOSED {self.last_order_type} price:{price} adj_price:{adj_price} qty:{self.qty} percentage_profit:{percentage_profit} absolute_profit:{self.absolute_profit} margin:{self.margin} fee:{_fee}')
        # sleep(10)

    # Execute one time step within the environment
    def step(self, action):
        self.last_order_type = ""
        self.absolute_profit = 0.0
        if self.in_position:
            high, low, close = self.df[self.current_step, 1:4]
            mark_close = self.df_mark[self.current_step, 4]
            self.in_position_counter += 1
            # change value for once in 8h, for 1m TF 8h=480
            if self.in_position_counter % 480 == 0:
                self.margin -= abs(self.qty) * close * self.funding_rate
            self._get_pnl(mark_close, update=True)
            if self._check_margin():
                self.liquidations += 1
                self._close_position(mark_close, liquidated=True)
            elif (self.stop_loss is not None) and (
                    ((low <= self.stop_loss_price) and (self.qty > 0))
                    or ((high >= self.stop_loss_price) and (self.qty < 0))
            ):
                self._close_position(self.stop_loss_price, sl=True)
            elif (self.take_profit is not None) and (
                    ((high >= self.take_profit_price) and (self.qty > 0))
                    or ((low <= self.take_profit_price) and (self.qty < 0))
            ):
                self._close_position(self.take_profit_price, tp=True)
            elif (action == 1 and self.qty < 0) or (action == 2 and self.qty > 0):
                self._close_position(close)
        elif action == 1:
            self._open_position("long", self.df[self.current_step, 3])
        elif action == 2:
            self._open_position("short", self.df[self.current_step, 3])
        else:
            self.with_gain_c += self.init_balance < self.balance + self.save_balance
        # self.info = {'action': action,
        #         'reward': 0,
        #         'step': self.current_step,
        #         'exec_time': time() - self.creation_t}
        return self._next_observation(), self.reward, self.done, False, self.info

    def _finish_episode(self):
        if self.in_position:
            self._close_position(self.enter_price)
        super()._finish_episode()
        if self.verbose:
            print(
                f" liquidations: {self.liquidations} liq_losses: ${self.liquidation_losses:.2f}"
            )

    def render(self, indicator_or_reward=None, visualize=False, *args, **kwargs):
        if visualize or self.visualize:
            if self.in_position:
                _balance = self.balance + (self.margin + (self.margin * self.pnl))
            else:
                _balance = self.balance
            if indicator_or_reward is None:
                indicator_or_reward = self.df[self.current_step, -1]
            render_row = [
                self.dates[self.current_step],
                *self.df[self.current_step, 0:4],
                indicator_or_reward,
                _balance,
            ]
            trade_info = [self.last_order_type, round(self.absolute_profit, 2)]
            #   print(f'trade_info {trade_info}')
            self.visualization.append(render_row, trade_info)
            self.visualization.render()
