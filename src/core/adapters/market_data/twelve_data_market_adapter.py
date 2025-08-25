from os import getenv
from dotenv import load_dotenv
from datetime import datetime
from dateutil.parser import parse
from typing import Optional

import requests

from src.core.models.bar import Bar
from src.core.models.asset import Asset, AssetType
from src.core.ports.market_data_port import MarketDataPort

START_DATE = None               # e.g., "2023-01-01"
END_DATE = None                 # e.g., "2023-06-01"
UNIT_PIP_VALUE = 30             # 1 unit = 30 pips
TIMEFRAME = "1h"                # 1-hour candles

class TwelveDataMarketAdapter(MarketDataPort):

    # Load environment variables from .env file
    load_dotenv()
    API_KEY = getenv("TWELVE_DATA_API_KEY")
    ENDPOINT = "https://api.twelvedata.com/time_series"

    def __init__(self, asset: Asset, output_size: int = 5000):

        self.asset = asset
        self.output_size = output_size
        self._list_data: list[Bar] = []
        self._counter: int = 0

    @property
    def current_bar(self):
        capped_last_bar_idx: int = min(self._counter, len(self._list_data)-1)
        return self._list_data[capped_last_bar_idx]

    def request_historical_data(self, asset: Asset, start_date: datetime, end_date: datetime):
        match asset.asset_type:
            case AssetType.FOREX:

                interval = TwelveDataMarketAdapter._get_duration_and_interval(start_date, end_date)

                params = {
                    "symbol": repr(asset),
                    "start_date": start_date,
                    "end_date": end_date,
                    "interval": interval,
                    "apikey": TwelveDataMarketAdapter.API_KEY,
                    "format": "JSON"
                }

                response = requests.get(TwelveDataMarketAdapter.ENDPOINT, params=params)
                response.raise_for_status()
                requested_data = response.json()

                if "values" not in requested_data:
                    raise ValueError(f"Invalid API response: {requested_data.get('message', requested_data)}")
                
                # Parse and clean data
                self._list_data = [
                        Bar(
                            timestamp=parse(bar["datetime"]), 
                            open=float(bar['open']), 
                            high=float(bar["high"]),
                            low=float(bar['low']),
                            close=float(bar['close']),
                            volume=None
                            ) for bar in reversed(requested_data["values"])
                ]


    def next_bar(self, asset: Asset) -> Optional[Bar]:
        if len(self._list_data) > self._counter:
            bar = self._list_data[self._counter]
            self._counter += 1
            return bar
        else:
            return None

    @classmethod
    def _get_duration_and_interval(cls, start_date: datetime, end_date: datetime):
        """
        Determine duration string and Twelve Data interval based on date range.

        Parameters:
            start_date (datetime): Beginning of historical data window
            end_date (datetime): End of historical data window

        Returns:
            tuple[str, str]: (duration, interval)
        """
        delta = end_date - start_date
        seconds = delta.total_seconds()

        if seconds < 60:
            interval = "1min"   # smallest Twelve Data interval is 1min

        elif seconds < 7 * 86400:  # less than 1 week
            interval = "1h"

        else:  # 1 year or more
            interval = "1day"

        return interval