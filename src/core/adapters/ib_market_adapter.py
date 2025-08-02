from src.core.ports.market_data_port import MarketDataPort
from src.core.models.asset import Asset
from src.core.models.bar import Bar
from src.core.services.ibapi_client import IBApi
from ibapi.contract import Contract
from datetime import datetime
import time

class IbapiDataAdapter(MarketDataPort):
    def __init__(self, ib_client: IBApi):
        self.ib_client = ib_client

    def _create_contract(self, asset: Asset) -> Contract:
        contract = Contract()
        contract.symbol = asset.symbol
        contract.secType = "CASH"
        contract.exchange = asset.exchange
        contract.currency = asset.currency
        return contract

    def request_historical_data(self, symbol: str, start_date: datetime, end_date: datetime):
        self.ib_client.historical_data_buffer = []
        self.ib_client.historical_data_done.clear()

        contract = self._create_contract(symbol)
        end_date_str = end_date.strftime("%Y%m%d %H:%M:%S")

        self.ib_client.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime=end_date_str,
            durationStr="1 M",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

        self.ib_client.historical_data_done.wait(timeout=10)

    def next_bar(self, symbol: str) -> Bar:
        bar_data = self.ib_client.historical_data_buffer[-1]
        return Bar(
            timestamp=bar_data.date,
            open=bar_data.open,
            high=bar_data.high,
            low=bar_data.low,
            close=bar_data.close,
            volume=bar_data.volume,
        )
