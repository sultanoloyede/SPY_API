"""
fetch_twelve_data.py
────────────────────
Fetch 1-hour OHLCV data for a given forex pair using the Twelve Data API.
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv
from poc.config import TIMEFRAME

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("TWELVE_DATA_API_KEY")
BASE_URL = "https://api.twelvedata.com/time_series"

def get_forex_data(pair: str, output_size: int = 5000) -> pd.DataFrame:
    """
    Fetch 1-hour historical OHLCV data for a forex pair.

    Parameters:
        pair (str): e.g., "EUR/USD"
        output_size (int): number of data points to fetch (default 5000)

    Returns:
        pd.DataFrame: datetime-indexed OHLCV dataframe
    """
    if not API_KEY:
        raise EnvironmentError("TWELVE_DATA_API_KEY not found in .env file")

    # Twelve Data uses symbol format like EUR/USD
    symbol = pair.upper().replace(" ", "")

    params = {
        "symbol": symbol,
        "interval": TIMEFRAME,
        "outputsize": output_size,
        "apikey": API_KEY,
        "format": "JSON"
    }

    print(f"Fetching {TIMEFRAME} data for {symbol}...")

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()

    if "values" not in data:
        raise ValueError(f"Invalid API response: {data.get('message', data)}")

    # Parse and clean data
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["datetime", "open", "high", "low", "close"]]
