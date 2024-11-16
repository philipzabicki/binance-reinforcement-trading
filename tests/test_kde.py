from time import time

from utils.data_collector import by_BinanceVision
from utils.feature_functions import *

TICKER = "BTCUSDT"
ITV = "5m"
MARKET_TYPE = "spot"
DATA_TYPE = "klines"

if __name__ == "__main__":
    df = by_BinanceVision(
        ticker=TICKER,
        interval=ITV,
        market_type=MARKET_TYPE,
        data_type=DATA_TYPE,
        split=False,
        delay=0,
    )

    # start_t = time()
    # profiler = cProfile.Profile()
    # profiler.enable()
    # df_w_kde = compute_probabilities_kde_separate(df_train)
    # profiler.disable()
    # profiler.print_stats(sort='cumtime')
    # print(f'compute_probabilities_kde_separate {time() - start_t}')

    start_t = time()
    # profiler = cProfile.Profile()
    # profiler.enable()
    df = candle_sizes_kde_pdf(df)
    # profiler.disable()
    # profiler.print_stats(sort='cumtime')
    print(f'compute_probabilities_kde_separate_parallel {time() - start_t}')

    df.to_csv("BTCUSDTspot_5m_modeling.csv", index=False)
    df.to_excel("BTCUSDTspot_5m_modeling.xlsx")
