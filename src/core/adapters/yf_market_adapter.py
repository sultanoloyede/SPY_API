from src.core.ports.market_data_port import MarketDataPort
from src.core.models.asset import Asset, AssetType
from src.core.models.bar import Bar
import yfinance as yf
import tempfile
from datetime import datetime, timedelta

import plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class YFMarketDataAdapter(MarketDataPort):


    def __init__(self, asset: Asset):
        self.asset = asset
        self._list_data:list[Bar] = []
        self._counter: int = 0

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
    def generate_data_plot(self):

        dates = [bar.timestamp for bar in self._list_data]
        opens = [bar.open for bar in self._list_data]
        highs = [bar.high for bar in self._list_data]
        lows = [bar.low for bar in self._list_data]
        closes = [bar.close for bar in self._list_data]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes
        ))
        fig.update_layout(
            title=f"Historical Data - {self.asset}", 
            xaxis_title="Date", 
            yaxis_title=f"{self.asset} Price ({self.asset.currency})",
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
            )
        py.offline.plot(fig)