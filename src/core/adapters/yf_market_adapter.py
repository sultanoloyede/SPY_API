from src.core.ports.market_data_port import MarketDataPort
from src.core.models.asset import Asset, AssetType
from src.core.models.bar import Bar
import yfinance as yf
import tempfile
from datetime import datetime, timedelta



class YFMarketDataAdapter(MarketDataPort):


    def __init__(self, asset: Asset):
        self.asset = asset
        self._list_data:list[Bar] = []
        self._counter: int = 0

    @property
    def current_bar(self):
        capped_last_bar_idx: int = min(self._counter, len(self._list_data)-1)
        return self._list_data[capped_last_bar_idx]

    def next_bar(self, asset: Asset) -> Bar:
        if len(self._list_data) > self._counter:
            bar = self._list_data[self._counter]
            self._counter += 1
            return bar
        else:
            return None

    def request_historical_data(self, asset: Asset, start_date: datetime, end_date: datetime):
        with tempfile.TemporaryDirectory() as tmpdir:
            if asset.asset_type == AssetType.FOREX: # We are dealing with forex
                stock = asset.symbol+asset.currency+"=X" # Write down the forex format for yfs
                data = yf.download(stock, start_date, end_date)
                self._list_data = [
                        Bar(
                            timestamp=index.to_pydatetime(), 
                            open=float(row['Open'].values[0]), 
                            high=float(row["High"].values[0]),
                            low=float(row['Low'].values[0]),
                            close=float(row['Close'].values[0]),
                            volume=float(row["Volume"].values[0])
                            ) for index, row in data.iterrows()
                ]
            else:
                raise NotImplemented("Asset Type not Implemented")