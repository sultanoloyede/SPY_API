from twelvedata import TDClient
import os
import time
import pandas as pd

# 1) Configure client 
td = TDClient(apikey=os.getenv("TWELVE_DATA_API_KEY", "c3c0a312b1674ab5a9bd0c7e6e39e1fa"))

# 2) Request 15 min candles for SPY
spy_candles = td.time_series(
    symbol = 'SPY',
    interval = '15min',
    outputsize = '5000'
)

# 3) Convert output to dataframe
df = spy_candles.as_pandas()
df.index = pd.to_datetime(df.index)
df_columns = ["open", "high", "low", "close", "volume"]

output_path = "/Users/bolajioloyede/Documents/FQ_Predictor/nba_betting_env/intraday/data/spy_15min.csv"
df.to_csv(output_path, index=True)


