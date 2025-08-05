import pytest
import pandas as pd
from poc.backtest.trade_result import TradeResult

def test_trade_result_instantiation():
    entry_time = pd.Timestamp("2024-01-01 10:00:00")
    exit_time = pd.Timestamp("2024-01-01 14:00:00")

    trade = TradeResult(
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=1.2000,
        exit_price=1.2060,
        direction="buy",
        result="win",
        rr_ratio=1.0,
        pips=60.0
    )

    assert trade.entry_time == entry_time
    assert trade.exit_time == exit_time
    assert trade.entry_price == 1.2000
    assert trade.exit_price == 1.2060
    assert trade.direction == "buy"
    assert trade.result == "win"
    assert trade.rr_ratio == 1.0
    assert trade.pips == 60.0
