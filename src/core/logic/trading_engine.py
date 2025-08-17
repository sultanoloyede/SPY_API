from src.core.ports.broker_trade_port import BrokerTradePort
from src.core.ports.market_data_port import MarketDataPort
from src.core.models.asset import Asset
from src.core.models.bar import Bar
from src.core.logic.strategy import Strategy

import threading
from typing import List

import plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TradingEngine:
    def __init__(self, broker: BrokerTradePort, market_data: MarketDataPort, strategies: List[Strategy]):
        self.broker = broker
        self.market_data = market_data
        self.strategies: List[Strategy] = strategies
        self.bar_data:List[Bar] = []

    def run(self, asset: Asset, threaded: bool = True):
        def engine_loop():
            while True:
                bar = self.market_data.next_bar(asset)  # Blocks until new bar is available in multithreaded applications
                if bar is None:
                    self.broker.compute_stats()
                    break
                self.bar_data.append(bar)
                for strategy in self.strategies:
                    strategy.evaluate(self.bar_data)

        if threaded:
            t = threading.Thread(target=engine_loop, daemon=True)
            t.start()
            return t
        else:
            engine_loop()

    def generate_data_plot(self):

        dates = [bar.timestamp for bar in self.market_data._list_data]
        opens = [bar.open for bar in self.market_data._list_data]
        highs = [bar.high for bar in self.market_data._list_data]
        lows = [bar.low for bar in self.market_data._list_data]
        closes = [bar.close for bar in self.market_data._list_data]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes
        ))
        fig.update_layout(
            title=f"Historical Data - {self.market_data.asset}", 
            xaxis_title="Date", 
            yaxis_title=f"{self.market_data.asset} Price ({self.market_data.asset.currency})",
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
            )
        py.offline.plot(fig)