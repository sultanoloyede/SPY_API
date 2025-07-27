import numpy as np
import pandas as pd
from math import pow, sqrt
from typing import List

from strategies.strategy import Strategy
from data_fusion.fuser import Fuser
from utils.config import INIT_VARIANCE

class KalmanFilter(Fuser):
    def __init__(self, strategies:List[Strategy], price_estimate_init:float=0, price_std_init:float=10000):
        self.strategies: List[Strategy] = strategies
        self.price_estimate: float = price_estimate_init # Agglomerate price estimate for the one future step
        self.price_std:float = price_std_init
    
    def predict(self):

        # Compute the Sum of the inverted ariances including the variance of the last estimate
        sum_inverse_variances: float = 1 / pow(self.price_std, 2) if self.price_std != 0 else 1 / INIT_VARIANCE # Set to small value in case it is 0.
        for strategy in self.strategies:
            sum_inverse_variances += 1 / pow(strategy.price_std, 2) if strategy.price_std != 0 else 1 / INIT_VARIANCE
        
        # Compute the Price Estimate and its std
        w_past: float = 1 / (pow(self.price_std, 2) * sum_inverse_variances)

        price_estimate_accumulation: float = w_past * self.price_estimate
        price_var_accumulation: float = pow(w_past, 2) * pow(self.price_std, 2)
        for strategy in self.strategies:
            wi = 1 / (pow(strategy.price_std, 2) * sum_inverse_variances) if strategy.price_std != 0 else 0
            price_estimate_accumulation += wi * strategy.price_estimate
            price_var_accumulation += pow(wi, 2) * pow(strategy.price_std, 2)
        
        # Set Computed values to field
        self.price_estimate = price_estimate_accumulation
        self.price_std = sqrt(price_var_accumulation)

