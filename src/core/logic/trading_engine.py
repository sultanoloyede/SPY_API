from src.core.ports.broker_trade_port import BrokerTradePort
from src.core.ports.market_data_port import MarketDataPort
from src.core.models.asset import Asset
from src.core.models.bar import Bar
from src.core.logic.strategy import Strategy
import threading
from typing import List

class TradingEngine:
    def __init__(self, broker: BrokerTradePort, market_data: MarketDataPort, strategies: List[Strategy]):
        self.broker = broker
        self.market_data = market_data
        self.strategies: List[Strategy] = strategies
        self.bar_data:List[Bar] = []

    def run(self, asset: Asset, threaded: bool = True):
        def engine_loop():
            while True:
                bar = self.market_data.next_bar(asset.symbol)  # Blocks until new bar is available in multithreaded applications
                if bar is None:
                    raise ValueError("Bar object not defined")  # Or break/raise if you want to handle end-of-stream
                self.bar_data.append(bar)
                for strategy in self.strategies:
                    strategy.evaluate(self.bar_data)

        if threaded:
            t = threading.Thread(target=engine_loop, daemon=True)
            t.start()
            return t
        else:
            engine_loop()

