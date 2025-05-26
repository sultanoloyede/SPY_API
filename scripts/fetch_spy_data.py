from dotenv import load_dotenv
import os
import time
import pandas as pd
from twelvedata import TDClient

# 0) Load .env into the environment
load_dotenv()  

# 1) Retrieve the key and validate
api_key = os.getenv("TWELVE_DATA_API_KEY")
if not api_key:
    raise RuntimeError("TWELVE_DATA_API_KEY not set. Please ensure you have a .env file with that variable.")

# 2) Configure the Twelve Data client
td = TDClient(apikey=api_key)

# 3) Request 15-minute candles for SPY
spy_candles = td.time_series(
    symbol="SPY",
    interval="15min",
    outputsize="5000"
)

# 4) Convert the output to a DataFrame
df = spy_candles.as_pandas()
df.index = pd.to_datetime(df.index)

# 5) Persist to CSV
output_path = os.getenv(
    "SPY_DATA_PATH",
    "/Users/bolajioloyede/Documents/FQ_Predictor/nba_betting_env/intraday/data/spy_15min.csv"
)
df.to_csv(output_path, index=True)

print(f"Wrote {len(df)} rows to {output_path}")
