
from core.ports.broker_trade_port import BrokerTradePort
from core.ports.market_data_port import MarketDataPort

class TradingEngine:
    def __init__(self, broker: BrokerTradePort, market_data: MarketDataPort):
        self.broker = broker
        self.market_data = market_data

    def run(self, symbol: str, start_date: str, end_date: str):
        pass