
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