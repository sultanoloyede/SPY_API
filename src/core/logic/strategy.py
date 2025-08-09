
from src.core.models.asset import Asset
from src.core.ports.broker_trade_port import BrokerTradePort
from src.core.ports.market_data_port import MarketDataPort
from src.core.models.bar import Bar
from typing import List

class Strategy:
    def __init__(self, broker: BrokerTradePort, asset: Asset):
        self.broker = broker
        self.asset = asset

    def evaluate(self, bar_data:List[Bar]):
        raise NotImplementedError("Subclasses must implement the evaluate method.")
