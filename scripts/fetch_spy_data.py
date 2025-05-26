from dotenv import load_dotenv
import os
import time
import pandas as pd
from twelvedata import TDClient
from pathlib import Path

# 1) Make sure the data folder exists (once at top of script)
os.makedirs("../data", exist_ok=True)

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

# 5) Save to CSV
df.to_csv("../data/spy_15min.csv", index=True)
