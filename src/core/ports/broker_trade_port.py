from abc import ABC, abstractmethod
from typing import Optional


class BrokerTradePort(ABC):

    @abstractmethod
    def buy(self, symbol: str, quantity: int, price: Optional[float] = None) -> str:
        pass

    @abstractmethod
    def sell(self, symbol: str, quantity: int, price: Optional[float] = None) -> str:
        pass

    @abstractmethod
    def bracket_order(self, symbol: str, quantity: int, entry_price: float,
                      take_profit: float, stop_loss: float) -> str:
        pass
