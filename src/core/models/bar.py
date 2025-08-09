from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass(frozen=True)
class Bar:
    timestamp: datetime = None
    open: float = None
    high: float = None
    low: float = None
    close: float = None
    volume: Optional[float] = 0.0
