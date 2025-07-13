from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *

import threading
import time
from datetime import datetime, timedelta
import pytz
import math
import ta_py as ta

import numpy as np
import pandas as pd

# IB Api Overwrite
class IBApi(EWrapper, EClient):

    orderId = 1

    def __init__(self, bot_callback):
        EClient.__init__(self, self)
        self.bot_callback = bot_callback
    
    # Gets Data for previous time window
    def historicalData(self, requestId, bar):
        try:
            bot.on_bar_update(requestId, bar, False)
        except Exception as e:
            print(e)
    
    # Gets Live data updates
    def historicalDataUpdate(self, requestId, bar):
        try:
            bot.on_bar_update(requestId, bar, False)
        except Exception as e:
            print(e)
    
    # Gets Historical data end
    def historicalDataEnd(self, requestId, start, end):
        print(requestId)
    
    def nextValidId(self, nextorderId):
        self.orderId = nextorderId
        
    def realtimeBar(self, reqId, barTime, open_, high, low, close, volume, wap, count):
        super().realtimeBar(reqId, barTime, open_, high, low, close, volume, wap, count)
        try:
            bot.on_bar_update(reqId, barTime, open_, high, low, close, volume, wap, count)
        except Exception as e:
            print(e)
    def error(self, id, errorCode, errorMsg, advancedOrderRejectJson=""):
        print(errorCode)
        print(errorMsg)

# Bar Object
class Bar:

    def __init__(self):
        self.open = 0
        self.low = 0
        self.high = 0
        self.close = 0
        self.volume = 0
        self.date = ''



# Bot Logic
class Bot:

    barSize = 1
    barsArray = []
    currentBar = Bar()
    smaPeriod = 50
    initialBarTime = datetime.now().astimezone(pytz.timezone("America/New_York"))
    contract = Contract()

    def __init__(self):
        # Create the IB API client and pass in the callback
        self.ib = IBApi(self.on_bar_update)

        # Connect to TWS or IB Gateway
        self.ib.connect("127.0.0.1", 7497, 1)

        # Start the network loop in a background thread
        ib_thread = threading.Thread(target=self.run_loop, daemon=True)
        ib_thread.start()

        # Delay for connection
        time.sleep(1)

        # Get symbol and bar size from user
        self.barSize = input("Enter the bar size: ").upper()
        symbol = input("Enter the base currency (e.g., EUR for EUR/USD): ").upper()
        quote = input("Enter the quote currency (e.g., USD for EUR/USD): ").upper()

        # Setting up min text
        mintext = " min"
        if (int(self.barSize) > 1):
            mintext = " mins"

        # Create contract
        self.contract.secType = "CASH"
        self.contract.symbol = symbol
        self.contract.exchange = "IDEALPRO"
        self.contract.currency = quote
        self.ib.reqIds(-1)
        
        self.ib.reqHistoricalData(self.ib.orderId, self.contract, "", "2 D", str(self.barSize)+mintext, "MIDPOINT", 1, True, True, [])


    def run_loop(self):
        self.ib.run()
    
    def bracketOrder(self, parentOrderId, action, quantity, profitTarget, stopLoss):
        # Create Parent Order
        parentOrder = Order()
        parentOrder.orderId = parentOrderId
        parentOrder.orderType = "MKT" # Buy at Market (MKT) price
        parentOrder.action = action
        parentOrder.totalQuantity = quantity
        parentOrder.transmit = False
        
        # Profit Target Order
        profitOrder = Order()
        profitOrder.orderId = parentOrderId + 1
        profitOrder.orderType = "LMT" # Buy at Limit (LMT) price
        profitOrder.lmtPrice = round(profitTarget, 2)
        profitOrder.action = "SELL"
        profitOrder.totalQuantity = quantity
        profitOrder.transmit = False
    
        # Stop Loss Order
        stopLossOrder = Order()
        stopLossOrder.orderId = parentOrderId + 2
        stopLossOrder.orderType = "STP" # Buy at Stop (STP) price
        stopLossOrder.lmtPrice = round(stopLoss, 2)
        stopLossOrder.action = "SELL"
        stopLossOrder.totalQuantity = quantity
        stopLossOrder.transmit = True

        bracketOrders = [parentOrder, profitOrder, stopLossOrder]
        return bracketOrders



    def on_bar_update(self, reqId, bar, realtime):
        if (realtime == False):
            self.barsArray.append(bar)
        else:
            bartime = datetime.strptime(bar.date, "%Y%m%d %H:%M:%S").astimezone(pytz.timezone("America/New_York"))
            minutes_diff = (bartime-self.initialBarTime).total_seconds() / 60.0
            self.currentBar.date = bartime

            # On Bar close
            if (minutes_diff > 0 and math.floor(minutes_diff) % self.barSize == 0):
                # Entry - If we have a higher high, a higher low and we cross the 50 SMA, we buy

                # Computing Simple Moving Average (SMA)
                closes = []
                for bar in self.barsArray:
                    closes.append(bar.close)
                self.close_array = pd.Series(np.asarray(closes))
                self.sma = ta.trend.sma(self.close_array, self.smaPeriod, True)
                print("SMA: " + str(self.sma[len(self.sma)-1]))

                # Calculate the higher highs and higher lows
                lastLow = self.barsArray[len(self.barsArray)-1].low
                lastHigh = self.barsArray[len(self.barsArray)-1].high
                lastClose = self.barsArray[len(self.barsArray)-1].close
                lastBar = self.barsArray[len(self.barsArray)-1]

                # Check criteria
                if (bar.close > lastHigh 
                    and self.currentBar.low > lastLow 
                    and bar.close > str(self.sma[len(self.sma)-1])
                    and lastClose < str(self.sma[len(self.sma)-2])):

                    # Bracket order of 2% Profit Target and 1% Stop Loss
                    profitTarget = bar.close * 1.02
                    stopLoss = bar.close*0.99
                    quantity = 25000 # Minimum value for ForEx

                    bracket = self.bracketOrder(self.ib.orderId, "BUY", quantity, profitTarget, stopLoss)

                    # Place bracket order
                    for order in bracket:
                        order.ocaGroup = "OCA_" + str(self.ib.orderId)
                        order.ocaType = 2
                        self.ib.placeOrder(order.orderId, self.contract, order)
                    
                    self.ib.orderId += 3

                    # Update the current bars
                    self.currentBar.close = bar.close
                    if self.currentBar.date != lastBar.date:
                        print("New Bar")
                        self.bars.append(self.currentBar)
                    self.currentBar.open = bar.open
        # Build Realtime bar
        if self.currentBar.open == 0:
            self.currentBar.open = bar.open
        if self.currentBar.high == 0 or bar.high > self.currentBar.high:
            self.currentBar.high = bar.high
        if self.currentBar.low == 0 or bar.low < self.currentBar.low:
            self.currentBar.low = bar.low


# Start bot
if __name__ == "__main__":
    bot = Bot()
