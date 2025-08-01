from abc import ABC, abstractmethod
from src.core.models.bar import Bar
from datetime import datetime

class MarketDataPort(ABC):

    @abstractmethod
    def next_bar(self, symbol: str) -> Bar:
        pass

    @abstractmethod
    def request_historical_data(self, symbol: str, start_date: datetime, end_date: datetime):
        pass
