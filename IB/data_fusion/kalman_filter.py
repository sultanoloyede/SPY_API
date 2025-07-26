import numpy as np
import pandas as pd
from math import pow, sqrt

from IB.strategies.strategy import Strategy
from IB.data_fusion.fuser import Fuser

class KalmanFilter(Fuser):
    def __init__(self, strategies:list[Strategy]):
        self.strategies: list[Strategy] = strategies
        self.price_estimate: float = 0 # Agglomerate price estimate for the one future step
        self.price_std:float = 0
    
    def predict(self):

        # Compute the Sum of Variances including the variance of the last estimate
        sum_of_variances: float = self.price_std # Set to 0 if price estimate of the pass is not important
        for strategy in self.strategies:
            sum_of_variances += pow(strategy.price_std, 2)
        
        # Compute the Price Estimate and its std
        w_past: float = 1 / (pow(self.price_std, 2) + sum_of_variances)

        price_estimate_accumulation: float = w_past * self.price_estimate
        price_var_accumulation: float = w_past * self.price_std
        for strategy in self.strategies:
            wi = 1 / (pow(strategy.price_std, 2) + sum_of_variances)
            price_estimate_accumulation += wi * strategy.price_estimate
            price_var_accumulation += pow(wi, 2) * pow(strategy.price_std, 2)
        
        # Set Computed values to field
        self.price_estimate = price_estimate_accumulation
        self.price_std = sqrt(price_var_accumulation)

