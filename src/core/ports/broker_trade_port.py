from abc import ABC, abstractmethod
from typing import Optional
from src.core.models.asset import Asset


class BrokerTradePort(ABC):

    @abstractmethod
    def buy(self, asset: Asset, quantity: int, price: Optional[float] = None) -> str:
        pass

    @abstractmethod
    def sell(self, sasset: Asset, quantity: int, price: Optional[float] = None) -> str:
        pass

    @abstractmethod
    def bracket_order(self, asset: Asset, quantity: int, entry_price: float,
                      take_profit: float, stop_loss: float) -> str:
        pass
