import pytest
from math import isclose, pow, sqrt
from IB.data_fusion.kalman_filter import KalmanFilter

# Mock Strategy class for testing
class MockStrategy:
    def __init__(self, price_estimate, price_std):
        self.price_estimate = price_estimate
        self.price_std = price_std

def test_kalman_filter_single_strategy():
    # We apply one strategy with a prior estimate
    s1 = MockStrategy(price_estimate=100, price_std=0.01)
    kf = KalmanFilter([s1], 110, 0.1)
    kf.predict()
    # With only one strategy, the result should be close to the input
    assert isclose(kf.price_estimate, 100, rel_tol=1e-2)
    assert isclose(kf.price_std, 0.0099, rel_tol=0.01)

def test_kalman_filter_two_strategies_equal_weight():
    # Two strategies with the same estimate and std
    s1 = MockStrategy(price_estimate=100, price_std=0.1)
    s2 = MockStrategy(price_estimate=100, price_std=0.1)
    kf = KalmanFilter([s1, s2], 100, 0.1)
    kf.predict()
    # The estimate should remain the same, std should decrease
    assert isclose(kf.price_estimate, 100, rel_tol=1e-6)
    assert kf.price_std < 0.1

def test_kalman_filter_two_strategies_different_estimates():
    # Two strategies with different estimates and stds
    s1 = MockStrategy(price_estimate=100, price_std=0.1)
    s2 = MockStrategy(price_estimate=110, price_std=0.2)
    kf = KalmanFilter([s1, s2], 105, 0.15)
    kf.predict()
    # The estimate should be between 100 and 110
    assert 100 < kf.price_estimate < 110
    # The std should be less than the smallest input std
    assert kf.price_std < 0.1

def test_kalman_filter_no_strategies():
    # No strategies, should keep prior
    kf = KalmanFilter([], 120, 0.5)
    kf.predict()
    assert isclose(kf.price_estimate, 120, rel_tol=1e-6)
    assert isclose(kf.price_std, 0.5, rel_tol=1e-6)
