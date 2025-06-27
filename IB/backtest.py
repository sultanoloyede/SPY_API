from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time


class IBApi(EWrapper, EClient):
    def __init__(self, bot_callback):
        EClient.__init__(self, self)
        self.bot_callback = bot_callback

    def realtimeBar(self, reqId, barTime, open_, high, low, close, volume, wap, count):
        bot.on_bar_update(reqId, barTime, open_, high, low, close, volume, wap, count)


# Bot Logic
class Bot:
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

        # Get symbol from user
        symbol = input("Enter the base currency (e.g., EUR for EUR/USD): ").upper()
        quote = input("Enter the quote currency (e.g., USD for EUR/USD): ").upper()

        # Create contract
        contract = Contract()
        contract.secType = "CASH"
        contract.symbol = symbol
        contract.exchange = "IDEALPRO"
        contract.currency = quote

        # Request real-time bars
        self.ib.reqRealTimeBars(
            reqId=0,
            contract=contract,
            barSize=5,             # 5 seconds
            whatToShow="MIDPOINT", # For Forex, use "MIDPOINT"
            useRTH=0,              # Set to 1 for Regular Trading Hours only
            realTimeBarsOptions=[]
        )

    def run_loop(self):
        self.ib.run()

    def on_bar_update(self, reqId, barTime, open_, high, low, close, volume, wap, count):
        print(f"[RealTimeBar] Time: {barTime}, Open: {open_}, High: {high}, Low: {low}, Close: {close}, Vol: {volume}, WAP: {wap}, Count: {count}")


# Start bot
if __name__ == "__main__":
    bot = Bot()
