from abc import ABC, abstractmethod
from typing import Optional

from src.core.models.bar import Bar
from src.core.models.asset import Asset


class BrokerTradePort(ABC):

    @abstractmethod
    def buy(self, asset: Asset, quantity: int) -> str:
        pass

    @abstractmethod
    def sell(self, sasset: Asset, quantity: int) -> str:
        pass

    @abstractmethod
    def bracket_order(self, asset: Asset, quantity: int, entry_price: float,
                      take_profit: float, stop_loss: float, action: str) -> str:
        pass
    
    @property
    @abstractmethod
    def value(self):
        pass

    @abstractmethod
    def compute_stats(self) -> None:
        pass