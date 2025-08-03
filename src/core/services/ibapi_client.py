# ibapi_client.py
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.common import *
from ibapi.contract import *
from threading import Thread
import threading
import time

class IBApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.nextOrderId = None
        self.order_id_ready = threading.Event()
        self.historical_data_buffer: list[BarData] = []
        self.historical_data_done = threading.Event()

    def nextValidId(self, orderId: int):
        self.nextOrderId = orderId
        self.order_id_ready.set()

    def get_next_order_id(self):
        self.order_id_ready.wait()
        order_id = self.nextOrderId
        self.nextOrderId += 1
        return order_id

    def historicalData(self, reqId, bar: BarData):
        self.historical_data_buffer.append(bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        self.historical_data_done.set()

    def connect_and_run(self, host="127.0.0.1", port=7497, client_id=1):
        self.connect(host, port, client_id)
        thread = Thread(target=self.run, daemon=True)
        thread.start()

        # Optional: Wait until connection is confirmed
        time.sleep(1)
        if self.isConnected():
            self.connected_event = True
            print("Connected to IB")
        else:
            print("Failed to connect to IB")

# Singleton instance holder
ibapi_singleton = None

def get_ibapi_client():
    global ibapi_singleton
    if ibapi_singleton is None:
        ibapi_singleton = IBApi()
        ibapi_singleton.connect_and_run()
    return ibapi_singleton
