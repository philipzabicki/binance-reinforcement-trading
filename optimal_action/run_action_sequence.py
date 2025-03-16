import pandas as pd
from numpy import inf

from environments.signal_env import SignalExecuteSpotEnv
from utils.get_data import by_BinanceVision

ACTION_DF = pd.read_csv(
    r"C:\Cloud\filips19mail\github\binance-reinforcement-trading\results_parts\final_combined_actions.csv"
)
# ACTION_DF.loc[ACTION_DF['Weight'] < 1.03, ['Weight', 'Action']] = [1.0, 0]
ACTION_SEQUENCE = ACTION_DF["Action"].tolist()
ACTION_DF["Opened"] = pd.to_datetime(ACTION_DF["Opened"])
ACTION_DF.set_index("Opened", inplace=True)

TICKER = "BTCUSDT"
ITV = "1h"
MARKET_TYPE = "spot"
DATA_TYPE = "klines"

ENV_KWARGS = {
    "max_steps": 0,
    "init_balance": 1_000,
    "no_action_finish": inf,
    "fee": 0.00075,
    "coin_step": 0.00001,
    "verbose": True,
    "visualize": False,
    "report_to_file": True,
}


def main():
    ohlcv = by_BinanceVision(
        ticker=TICKER,
        interval=ITV,
        market_type=MARKET_TYPE,
        data_type=DATA_TYPE,
        start_date="2018-01-01 00:00:00",
        end_date="2025-01-01 00:00:00",
        split=False,
        delay=0,
    )
    ohlcv["Opened"] = pd.to_datetime(ohlcv["Opened"])
    ohlcv.set_index("Opened", inplace=True)
    print(f"Dane OHLCV załadowane: {ohlcv.shape[0]} wierszy")

    start_time = ACTION_DF.index.min()
    end_time = ACTION_DF.index.max()

    ohlcv_part = ohlcv.loc[start_time:end_time].reset_index()
    env = SignalExecuteSpotEnv(df=ohlcv_part, **ENV_KWARGS)
    env.reset()
    env.signals = ACTION_SEQUENCE
    env()


if __name__ == "__main__":
    main()
