
from enum import Enum
from typing import Optional
from dataclasses import dataclass

class AssetType(Enum):
    FOREX = 1
    STOCK = 2

@dataclass
class Asset:

    asset_type: AssetType
    symbol: str
    currency: str

    def __repr__(self):
        match self.asset_type:
            case AssetType.FOREX:
                return f"{self.symbol}/{self.currency}"
            case AssetType.STOCK:
                pass