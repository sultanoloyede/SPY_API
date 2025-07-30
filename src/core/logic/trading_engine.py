
from src.core.ports.broker_trade_port import BrokerTradePort
from src.core.ports.market_data_port import MarketDataPort
from src.core.models.asset import Asset

class TradingEngine:
    def __init__(self, broker: BrokerTradePort, market_data: MarketDataPort):
        self.broker = broker
        self.market_data = market_data

    def run(self, asset: Asset, start_date: str, end_date: str):
        pass