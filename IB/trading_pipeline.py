import time
import threading
from datetime import datetime
from typing import Dict, Optional
import pandas as pd

try:
    from ibapi.client import EClient # Client sends information to TWS
    from ibapi.wrapper import EWrapper # Wrapper handles callbacks from TWS
    from ibapi.contract import Contract
    from ibapi.order import Order
    from ibapi.common import BarData
except ModuleNotFoundError as error:
    print("Ensure that IB Api was downloaded using the latest release found on the github and then downloaded using pip.")
    raise
from strategies.strategy import Strategy
from strategies.mean_reversion import MeanReversionStrategy


class IBApi(EWrapper, EClient):
    """
    IBApi is a strategy manager class that integrates with the Interactive Brokers (IB) API using a publisher-observer architecture. 
    It inherits from both EWrapper and EClient to handle both client requests and server responses.
    """

    #-----------------#Publisher Method Definitions#-----------------#

    def register(self, strategy: Strategy):
        self.strategies.append(strategy)
        strategy.start()

    def unregister(self, strategy: Strategy):
        strategy.stop()
        strategy.join()
        self.strategies.remove(strategy)

    def notify_all(self, data):
        for strategy in self.strategies:
            strategy.on_new_data(data)

    #-----------------#IBApi Definitions#-----------------#
    @staticmethod
    def get_forex_contract(base:str, quote:str) -> Contract:
        contract = Contract()
        contract.symbol = base
        contract.currency = quote
        contract.secType = "CASH"
        contract.exchange = "IDEALPRO"
        return contract
    
    def __init__(self):
        EClient.__init__(self, self)
        self.data: Dict[int, pd.DataFrame] = {} # Pandas dataframe containing the live updating bar timeseries
        self.strategies:list[Strategy] = [] # List containing all the registered strategies
        self.data_ready:threading.Event = threading.Event()

    # Error method, called whenever an error occurs within IB
    def error(self, reqId: int, errorCode: int, errorString: str, advanced: any=None) -> None:
        print(f"Error: {reqId}, Code: {errorCode}, Msg: {errorString}")
    
    # Gathers historical data 1 day of 1 minute bars ending now
    #NOTE: This method calls uses the client to request historical data. The result needs to 
    #      be processed with the wrapper and the historicalData callback 
    def initialize_data(self, reqId:int, contract:Contract) ->pd.DataFrame:
        self.data[reqId] = pd.DataFrame(columns=["time", "high", "low", "close"])
        self.data[reqId].set_index("time", inplace=True)

        # Fetch historical data using client
        self.data_ready.clear()
        self.reqHistoricalData(
            reqId=reqId,
            contract=contract,
            endDateTime='',
            durationStr='1 D', # Ask for past data for 1 Day
            barSizeSetting='1 min', # Ask for 1 minute bar data interval
            whatToShow='MIDPOINT',
            useRTH=0,
            formatDate=2, # We use unix timestamp to do numpy math
            keepUpToDate=True,
            chartOptions=[]
        )
        if not self.data_ready.wait(timeout=10): # We wait until the client has processed this task else we 
            raise TimeoutError(f"Historical data request timed out for reqId{reqId}")
        return self.data[reqId]
    
    # Historial Data call back. Is called whenever TWS returns historical data
    def historicalData(self, reqId:int, bar:BarData) -> None:
        df = self.data[reqId]
        df.loc[
            pd.to_datetime(int(bar.date), unit="s"),
            ["high", "low", "close"]
        ] = [bar.high, bar.low, bar.close]
        df = df.astype(float)
        self.data[reqId] = df

    # Callback for live updating data
    def historicalDataUpdate(self, reqId: int, bar: BarData) -> None:
        df = self.data[reqId]
        timestamp = pd.to_datetime(int(bar.date), unit="s")

        # Update last row
        if timestamp in df.index:
            df.loc[timestamp] = [bar.high, bar.low, bar.close]
        else:
            df.loc[timestamp] = [bar.high, bar.low, bar.close]

        df = df.astype(float)
        self.data[reqId] = df
    
    # Callback for once live streamed data has been completed
    def historicalDataEnd(self, reqId, start, end):
        self.data_ready.set() 
        self.notify_all(self.data[reqId])

    # Creates a bracket order with a market entry, profit target, and stop loss
    def bracketOrder(self, parentOrderId, action, quantity, profitTarget, stopLoss):
        parentOrder = Order()
        parentOrder.orderId = parentOrderId
        parentOrder.orderType = "MKT"
        parentOrder.action = action
        parentOrder.totalQuantity = quantity
        parentOrder.transmit = False

        profitOrder = Order()
        profitOrder.orderId = parentOrderId + 1
        profitOrder.orderType = "LMT"
        profitOrder.lmtPrice = round(profitTarget, 5)  # Forex uses 5 decimal precision
        profitOrder.action = "SELL" if action == "BUY" else "BUY"
        profitOrder.totalQuantity = quantity
        profitOrder.transmit = False

        stopLossOrder = Order()
        stopLossOrder.orderId = parentOrderId + 2
        stopLossOrder.orderType = "STP"
        stopLossOrder.auxPrice = round(stopLoss, 5)
        stopLossOrder.action = "SELL" if action == "BUY" else "BUY"
        stopLossOrder.totalQuantity = quantity
        stopLossOrder.transmit = True

        return [parentOrder, profitOrder, stopLossOrder]



if __name__ == "__main__":
    ib = IBApi()
    ib.connect("127.0.0.1", 7497, clientId=1)
    threading.Thread(target=ib.run, daemon=True).start()

    eur_usd_contract = IBApi.get_forex_contract("EUR", "USD")
    mrs = MeanReversionStrategy(ib, eur_usd_contract)
    
    ib.register(mrs)

    data = ib.initialize_data(99, eur_usd_contract)