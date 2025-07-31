
from src.core.ports.broker_trade_port import BrokerTradePort
from src.core.ports.market_data_port import MarketDataPort
from src.core.models.asset import Asset
from src.core.models.bar import Bar
from src.core.logic.strategy import Strategy
import threading

class TradingEngine:
    def __init__(self, broker: BrokerTradePort, market_data: MarketDataPort, strategies: list[Strategy]):
        self.broker = broker
        self.market_data = market_data
        self.strategies: list[Strategy] = strategies
        self.bar_data:list[Bar]

    def run(self, asset: Asset, threaded: bool = True):
        def engine_loop():
            while True:
                bar = self.market_data.next_bar(asset.symbol)  # Blocks until new bar is available
                if bar is None:
                    raise ValueError("Bar object not defined")  # Or break/raise if you want to handle end-of-stream
                self.bar_data.append(bar)
                for strategy in self.strategies:
                    strategy.evaluate()

        if threaded:
            t = threading.Thread(target=engine_loop, daemon=True)
            t.start()
            return t
        else:
            engine_loop()

