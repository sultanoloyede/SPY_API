import numpy as np
import pandas as pd
from typing import Tuple
from src.core.logic.strategy import Strategy
from src.core.models.asset import Asset, AssetType
from src.core.models.bar import Bar
from src.core.ports.broker_trade_port import BrokerTradePort
from src.utils.logger import logger


class MovingAverageCrossoverStrategy(Strategy):

    def __init__(self, broker, asset):
        super().__init__(broker, asset)
        self.data_50_sma = []
        self.data_200_sma = []

    def evaluate(self, bar_data:list[Bar]):
        """
        Evaluate the strategy using the latest historical data from the market data port.
        Executes trades via the broker port if a crossover is detected.
        """

        # Compute the current 50 day moving average
        if len(bar_data) > 200:
            accumulation_50_day = 0
            accumulation_200_day = 0

            # Iterate over first 50 bars
            for i in range(50):
                bar = bar_data[i]
                # TODO: Research which candlestick info we should use between OCHL
                accumulation_50_day += bar.high
                accumulation_200_day += bar.high

            # Iterate over next 150 bars
            for i in range(51, 200):
                bar = bar_data[i]
                # TODO: Research which candlestick info we should use between OCHL
                accumulation_200_day += bar.high

            sma_50_day = accumulation_50_day / 50
            sma_200_day = accumulation_200_day / 200

            # Update the 50 day SMA tally
            self.data_50_sma.append(sma_50_day)
            while len(self.data_50_sma) > 50:
                self.data_50_sma.pop(0)
          
            # Update the 200 day SMA tally
            self.data_200_sma.append(sma_200_day)
            while len(self.data_200_sma) > 200:
                self.data_200_sma.pop(0)

            # Compute crossing
            if (
                len(self.data_50_sma) > 1 and len(self.data_200_sma) > 1 and
                self.data_50_sma[-1] < self.data_200_sma[-1] and # Death Crossing condition
                self.data_50_sma[-2] >= self.data_200_sma[-2]
                ):
                logger.info("Detected Death Cross of 50/200 bar SMA")
                self.broker.sell(self.asset, 1)
            elif (
                len(self.data_50_sma) > 1 and len(self.data_200_sma) > 1 and
                self.data_50_sma[-1] > self.data_200_sma[-1] and # Golden Crossing condition
                self.data_50_sma[-2] <= self.data_200_sma[-2]
                ):
                logger.info("Detected Golden Cross of 50/200 bar SMA")
                self.broker.buy(self.asset, 1)
            else:
                pass
        else:
            logger.info(f"Insufficient ammount of data to compute 50 and 200 bar moving average. Only {len(bar_data)} bars available.")