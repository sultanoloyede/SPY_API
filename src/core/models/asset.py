
from enum import Enum
from typing import Optional
from dataclasses import dataclass

@dataclass
class Asset:

    asset_type: str
    symbol: str