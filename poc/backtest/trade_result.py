from dataclasses import dataclass
import pandas as pd

@dataclass
class TradeResult:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: str  # "buy" or "sell"
    result: str     # "win" or "loss"
    rr_ratio: float
    pips: float     # Positive for win, negative for loss
