from environments.spot_rl_env import SpotTakerRL
from numpy import inf

from utils.data_collector import by_BinanceVision

TICKER = "BTCUSDT"
ITV = "5m"
MARKET_TYPE = "spot"
DATA_TYPE = "klines"
TRADE_START_DATE = "2023-12-04"
TRADE_END_DATE = "2024-06-01"
# Better to take more previous data for some TA features
DF_START_DATE = "2023-09-04"
DF_END_DATE = "2024-06-01"
ENV_KWARGS = {
    "lookback_size": 10,
    "max_steps": 8_640,  # 30 days in 5m intervals
    # 'start_date': TRADE_START_DATE,
    # 'end_date': TRADE_END_DATE,
    "init_balance": 1_000,
    "no_action_finish": inf,
    "fee": 0.00075,
    "coin_step": 0.00001,
    "stop_loss": 0.03,
    "take_profit": 0.015,
    "visualize": False,
    "render_range": 60,
    # 'slippage': get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
    "verbose": True,
    "report_to_file": True,
}

if __name__ == "__main__":
    df = by_BinanceVision(
        ticker=TICKER,
        interval=ITV,
        market_type=MARKET_TYPE,
        data_type=DATA_TYPE,
        # start_date=DF_START_DATE,
        # end_date=DF_END_DATE,
        split=False,
        delay=0,
    )
    print(f"df used: {df}")

    test_env = SpotTakerRL(df=df, **ENV_KWARGS)
    print(test_env.reset())
    terminated = False
    while not terminated:
        action = test_env.action_space.sample()
        # action = random.choice([0, 1, 2], p=[0.7, 0.25, 0.05])
        # print(f"ACTION {action}")
        observation, reward, terminated, info = test_env.step(action)
        # test_env.render()
        # print(observation, reward, terminated, info)
