from src.core.logic.strategy import Strategy
from src.core.models.asset import Asset, AssetType
from src.core.models.bar import Bar
from src.core.ports.broker_trade_port import BrokerTradePort
from src.utils.logger import logger


class MovingAverageCrossoverStrategy(Strategy):

    def __init__(self, broker, asset, fast: int = 50, slow: int = 200):
        super().__init__(broker, asset)

        if fast >= slow:
            raise ValueError("Fast period must be strictly less than slow period.")

        self.fast = fast
        self.slow = slow
        self.data_fast_sma = []
        self.data_slow_sma = []

    def evaluate(self, bar_data: list[Bar]):

        if len(bar_data) > self.slow:

            last_fast_bars = bar_data[-self.fast:]
            sma_fast = sum(bar.high for bar in last_fast_bars) / self.fast


            last_slow_bars = bar_data[-self.slow:]
            sma_slow = sum(bar.high for bar in last_slow_bars) / self.slow

            self.data_fast_sma.append(sma_fast)
            while len(self.data_fast_sma) > self.fast:
                self.data_fast_sma.pop(0)

            self.data_slow_sma.append(sma_slow)
            while len(self.data_slow_sma) > self.slow:
                self.data_slow_sma.pop(0)

            current_bar = bar_data[-1]
            current_price = current_bar.low
            current_date = current_bar.timestamp

            if (
                len(self.data_fast_sma) > 1 and len(self.data_slow_sma) > 1 and
                self.data_fast_sma[-1] < self.data_slow_sma[-1] and  # Death Cross
                self.data_fast_sma[-2] >= self.data_slow_sma[-2]
            ):
                logger.info(
                    f"Detected Death Cross of {self.fast}/{self.slow} bar SMA at {current_date}"
                )
                self.broker.sell(self.asset, 1, current_price)

            elif (
                len(self.data_fast_sma) > 1 and len(self.data_slow_sma) > 1 and
                self.data_fast_sma[-1] > self.data_slow_sma[-1] and  # Golden Cross
                self.data_fast_sma[-2] <= self.data_slow_sma[-2]
            ):
                logger.info(
                    f"Detected Golden Cross of {self.fast}/{self.slow} bar SMA at {current_date}"
                )
                self.broker.buy(self.asset, 1, current_price)

            else:
                pass
        else:
            logger.debug(
                f"Insufficient ammount of data to compute {self.fast} and {self.slow} "
                f"bar moving average. Only {len(bar_data)} bars available."
            )
