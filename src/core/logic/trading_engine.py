from src.utils.logger import logger
from src.core.ports.broker_trade_port import BrokerTradePort
from src.core.ports.market_data_port import MarketDataPort
from src.core.models.asset import Asset
from src.core.models.bar import Bar
from src.core.logic.strategy import Strategy
from src.utils.config import RISK_FREE_RATE

import threading
from datetime import datetime

import plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TradingEngine:
    def __init__(self, broker: BrokerTradePort, market_data: MarketDataPort, strategies: list[Strategy]):
        self.broker = broker
        self.market_data = market_data
        self.strategies: list[Strategy] = strategies
        self.bar_data:list[Bar] = []
        self.portfolio_value: dict[datetime, int] = {}

    def run(self, asset: Asset, threaded: bool = True):
        def engine_loop():
            while True:

                # Blocks until new bar is available in multithreaded applications
                bar = self.market_data.next_bar(asset)
                
                # When market data is completed, compute statistics and end this function
                if bar is None:
                    self.broker.compute_stats()
                    break

                # Add bar to list data and update portfolio value
                self.bar_data.append(bar)
                self.portfolio_value[bar.timestamp] = self.broker.value

                # Iterate over strategies to generate signals
                for strategy in self.strategies:
                    strategy.evaluate(self.bar_data)

        if threaded:
            t = threading.Thread(target=engine_loop, daemon=True)
            t.start()
            return t
        else:
            engine_loop()