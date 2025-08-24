from src.core.ports.market_data_port import MarketDataPort
from src.core.models.asset import Asset, AssetType
from src.core.models.bar import Bar
from src.core.services.ibapi_client import IBApi
from src.utils.logger import logger
from ibapi.contract import Contract
from ibapi.common import BarData as IBDATA
from datetime import datetime
from queue import Queue
import pytz
from ibapi.account_summary_tags import AccountSummaryTags

class IbApiDataAdapter(MarketDataPort):
    def __init__(self, ib_client: IBApi):
        self.ib_client = ib_client
        self._list_data: list[Bar] = []

    def _create_contract(self, asset: Asset) -> Contract:
        if asset.asset_type == AssetType.FOREX:
            contract = Contract()
            contract.symbol = asset.symbol
            contract.secType = "CASH"
            contract.exchange = "IDEALPRO"
            contract.currency = asset.currency
            return contract
        else:
            logger.error("Contract creation failed: Asset type not implemented.")
    
    def _get_ib_duration_and_bar_size(self, start_date: datetime, end_date: datetime):
        delta = end_date - start_date
        seconds = delta.total_seconds()

        if seconds < 60:
            duration = f"{max(int(seconds), 30)} S"
            bar_size = "30 secs"

        elif seconds < 7 * 86400:  # less than 1 week
            days = max(int(seconds // 86400), 1)
            duration = f"{days} D"
            bar_size = "1 hour"

        elif seconds < 30 * 86400:  # less than ~1 month
            weeks = max(int(seconds // (7 * 86400)), 1)
            duration = f"{weeks} W"
            bar_size = "1 day"

        elif seconds < 365 * 86400:  # less than 1 year
            months = max(int(seconds // (30 * 86400)), 1)
            duration = f"{months} M"
            bar_size = "1 day"

        else:  # 1 year or more
            years = max(int(seconds // (365 * 86400)), 1)
            duration = f"{years} Y"
            bar_size = "1 week"

        return duration, bar_size
    
    def request_historical_data(self, asset: Asset, start_date: datetime, end_date: datetime = None):

        contract = self._create_contract(asset)

        if end_date is not None: # Historical Update
        
            end_date_utc = end_date.astimezone(pytz.utc)
            end_date_str = end_date_utc.strftime("%Y%m%d-%H:%M:%S")
            self._duration, self._bar_size = self._get_ib_duration_and_bar_size(start_date, end_date)
        
            self.ib_client.reqHistoricalData(
                reqId=1,
                contract=contract,
                endDateTime=end_date_str,
                durationStr=self._duration,
                barSizeSetting=self._bar_size, 
                whatToShow='MIDPOINT',
                useRTH=0,
                formatDate=2, # We use unix timestamp to do numpy math
                keepUpToDate=False,
                chartOptions=[]
            )

        elif end_date is None: # Live updates
            
            end_date = datetime.today()
            end_date_utc = end_date.astimezone(pytz.utc)
            end_date_str = end_date_utc.strftime("%Y%m%d-%H:%M:%S")
            self._duration, self._bar_size = self._get_ib_duration_and_bar_size(start_date, end_date)
            
            self.ib_client.reqHistoricalData(
                reqId=1,
                contract=contract,
                endDateTime='', # IB assumes that empty string means till "current present moment"
                durationStr=self._duration, # Ask for past data for 1 Day
                barSizeSetting=self._bar_size, # Ask for 1 minute bar data interval
                whatToShow='MIDPOINT',
                useRTH=0,
                formatDate=2, # We use unix timestamp to do numpy math
                keepUpToDate=True,
                chartOptions=[]
            )
            # Updated account balance asynchronously
            self.ib_client.reqAccountSummary(self.ib_client.nextOrderId, "All", AccountSummaryTags.AllTags)
    
    @property
    def current_bar(self) -> Bar:
        return self._list_data[-1]

    def next_bar(self, asset: Asset) -> Bar:
        self._list_data.append(self.ib_client.historical_data_buffer.get())
        return self._list_data[-1]
