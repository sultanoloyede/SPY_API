import pytest
from abc import ABCMeta
from poc.strategies.base_strategy import BaseStrategy

def test_base_strategy_is_abstract():
    """
    Ensures BaseStrategy is abstract and has required method.
    """
    assert isinstance(BaseStrategy, ABCMeta) or hasattr(BaseStrategy, '__abstractmethods__')
    assert "generate_signals" in dir(BaseStrategy)
