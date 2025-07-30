from abc import ABC, abstractmethod
from core.models.bar import Bar

class MarketDataPort(ABC):
    @abstractmethod
    def next_bar(self, symbol: str) -> Bar:
        pass

    @abstractmethod
    def request_historical_data(self, symbol: str, start_date: str, end_date: str):
        pass
