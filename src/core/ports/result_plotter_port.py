from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Type
from src.core.models.bar import Bar


class ResultPlotterPort(ABC):
    
    
    @classmethod
    @abstractmethod
    def plot(
        cls: Type["ResultPlotterPort"],
        bar_data: List[Bar],
        portfolio_value: Dict[datetime, int],
        asset_name: str,
        currency: str,
    ) -> None:
        pass
