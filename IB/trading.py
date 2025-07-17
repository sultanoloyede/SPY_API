from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *

import threading
import time
from datetime import datetime
import pytz
import math
import ta
import dateutil

import numpy as np
import pandas as pd

# Bar Object
class Bar:
    def __init__(self):
        self.open = 0
        self.low = 0
        self.high = 0
        self.close = 0
        self.volume = 0
        self.date = ''


# IB API Overwrite
class IBApi(EWrapper, EClient):
    orderId = 1

    def __init__(self, bot_callback):
        EClient.__init__(self, self)
        self.bot_callback = bot_callback

    def historicalData(self, requestId, bar):
        try:
            bot.on_bar_update(requestId, bar, False)
        except Exception as e:
            print(e)

    def historicalDataUpdate(self, requestId, bar):
        try:
            bot.on_bar_update(requestId, bar, True)
        except Exception as e:
            print(e)

    def historicalDataEnd(self, requestId, start, end):
        print(f"Request ID: {requestId}")

    def nextValidId(self, nextorderId):
        self.orderId = nextorderId

    def error(self, id, errorCode, errorMsg, advancedOrderRejectJson=""):
        print(errorCode)
        print(errorMsg)


# Bot Logic
class Bot:
    def __init__(self):
        self.barSize = 1
        self.barsArray = []
        self.currentBar = Bar()
        self.smaPeriod = 10
        self.initialBarTime = datetime.now().astimezone(pytz.timezone("US/Eastern"))
        self.contract = Contract()

        self.ib = IBApi(self.on_bar_update)
        self.ib.connect("127.0.0.1", 7497, 1)

        ib_thread = threading.Thread(target=self.run_loop, daemon=True)
        ib_thread.start()

        time.sleep(1)

        self.barSize = 1 # int(input("Enter the bar size: "))
        symbol = "USD" # input("Enter the base currency (e.g., EUR for EUR/USD): ").upper()
        quote = "CAD" # input("Enter the quote currency (e.g., USD for EUR/USD): ").upper()

        mintext = " mins" if self.barSize > 1 else " min"

        self.contract.secType = "CASH"
        self.contract.symbol = symbol
        self.contract.exchange = "IDEALPRO"
        self.contract.currency = quote

        self.ib.reqIds(-1)
        self.ib.reqHistoricalData(self.ib.orderId, self.contract, "", "2 D", f"{self.barSize}{mintext}", "MIDPOINT", 1, True, True, [])

    def run_loop(self):
        self.ib.run()

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
        profitOrder.lmtPrice = round(profitTarget, 2)
        profitOrder.action = "SELL"
        profitOrder.totalQuantity = quantity
        profitOrder.transmit = False

        stopLossOrder = Order()
        stopLossOrder.orderId = parentOrderId + 2
        stopLossOrder.orderType = "STP"
        stopLossOrder.lmtPrice = round(stopLoss, 2)
        stopLossOrder.action = "SELL"
        stopLossOrder.totalQuantity = quantity
        stopLossOrder.transmit = True

        return [parentOrder, profitOrder, stopLossOrder]

    def on_bar_update(self, reqId, bar, realtime):
        if not realtime:
            self.barsArray.append(bar)
            return
        
        parts = bar.date.split(' ')
        bartime = datetime.strptime(parts[0]+' ' + parts[1], "%Y%m%d %H:%M:%S").replace(tzinfo=dateutil.tz.gettz(parts[2]))
        minutes_diff = (bartime-self.initialBarTime).total_seconds() / 60.0
        self.currentBar.date = bartime


        if minutes_diff > 0 and math.floor(minutes_diff) % self.barSize == 0:
            self.currentBar.close = bar.close
            self.currentBar.volume = bar.volume
            self.barsArray.append(self.currentBar)
            self.currentBar = Bar()

            closes = [b.close for b in self.barsArray]
            close_array = pd.Series(np.asarray(closes))
            sma = ta.trend.SMAIndicator(close_array, self.smaPeriod, fillna=True).sma_indicator()

            if len(self.barsArray) < 2:
                return

            lastBar = self.barsArray[-2]
            lastLow = lastBar.low
            lastHigh = lastBar.high
            lastClose = lastBar.close

            print(f"bar.close: {bar.close}, lastHigh: {lastHigh}")
            print(f"self.currentBar.low: {self.currentBar.low}, lastLow: {lastLow}")
            print(f"bar.close: {bar.close}, SMA[-1]: {sma.iloc[-1]}")
            print(f"lastClose: {lastClose}, SMA[-2]: {sma.iloc[-2]}")
            print("SMA: " + str(sma.iloc[-1]))

            if (bar.close > lastHigh and
                self.currentBar.low > lastLow and
                bar.close > sma.iloc[-1] and
                lastClose < sma.iloc[-2]):

                profitTarget = bar.close * 1.02
                stopLoss = bar.close * 0.99
                quantity = 25000
                print("Placing Order")
                bracket = self.bracketOrder(self.ib.orderId, "BUY", quantity, profitTarget, stopLoss)
                print("Orders Placed")

                for order in bracket:
                    order.ocaGroup = "OCA_" + str(self.ib.orderId)
                    order.ocaType = 2
                    self.ib.placeOrder(order.orderId, self.contract, order)

                self.ib.orderId += 3

        # Build current bar
        if self.currentBar.open == 0:
            self.currentBar.open = bar.open
            self.currentBar.high = bar.high
            self.currentBar.low = bar.low
        else:
            self.currentBar.high = max(self.currentBar.high, bar.high)
            self.currentBar.low = min(self.currentBar.low, bar.low)


if __name__ == "__main__":
    bot = Bot()
