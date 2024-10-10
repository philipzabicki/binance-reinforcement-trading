from utils.data_collector import by_BinanceVision

TICKER = "BTCUSDT"
ITV = "5m"
MARKET_TYPE = "um"
DATA_TYPE = "klines"

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
    print(f"df used: {df.describe()}")
    df_mark = by_BinanceVision(
        ticker=TICKER,
        interval=ITV,
        market_type=MARKET_TYPE,
        data_type="markPriceKlines",
        # start_date=DF_START_DATE,
        # end_date=DF_END_DATE,
        split=False,
        delay=0,
    )
    print(f"df_mark used: {df_mark.describe()}")
