from unittest.mock import MagicMock, patch
from src.core.logic.trading_engine import TradingEngine
from src.core.models.bar import Bar
from datetime import datetime
import threading
import time
import pytest

@pytest.fixture
def mock_market_data():
    mock = MagicMock()
    mock.next_bar.side_effect = [
        Bar(open=100, high=110, low=90, close=105, volume=1000, timestamp=datetime(2025, 7, 31, 14, 30, 13, 102)),
        None  # Simulate end of stream
    ]
    return mock

@pytest.fixture
def mock_market_data_blocking():
    block_event = threading.Event()

    def blocking_next_bar():
        block_event.wait(timeout=1)  # Simulate blocking behavior
        return Bar(open=100, high=110, low=90, close=105, volume=1000, timestamp=datetime.now())

    mock = MagicMock()
    mock.next_bar.side_effect = blocking_next_bar
    mock._block_event = block_event  # Attach event for test control
    return mock

@pytest.fixture
def mock_broker():
    return MagicMock()

@pytest.fixture
def mock_strategy():
    return MagicMock()

@pytest.fixture
def mock_forex_asset():
    asset = MagicMock()
    asset.asset_type = "CASH"
    asset.symbol = "EUR.USD"
    return asset

def test_feed_next_bar(mock_broker, mock_market_data, mock_strategy, mock_forex_asset):
    trading_engine = TradingEngine(mock_broker, mock_market_data, [mock_strategy])

    trading_engine.run(asset=mock_forex_asset, threaded=False)

    # Assert that the strategy was called in with data
    mock_strategy.evaluate.assert_called_once()
    assert type(mock_strategy.evaluate.call_args[0][0]) == list
    assert type(mock_strategy.evaluate.call_args[0][0][0]) == Bar
