from abc import ABC, abstractmethod
from threading import Thread, Event
from typing import Tuple
import queue

class Strategy(ABC, Thread):
    """
    Abstract defintion for a strategy observer class.
    Sets up a thread-safe queue for data communication and an event to signal stopping the strategy.
    """
    def __init__(self):
        super().__init__()
        self.data_queue = queue.Queue()  # Thread-safe queue
        self.stop_event = Event()
        self.price_estimate: float = None
        self.price_std:float = None


    def on_new_data(self, data):
        self.data_queue.put(data)

    @abstractmethod
    def evaluate(self, data) -> Tuple[float, float]:
        pass

    def run(self):
        while not self.stop_event.is_set():
            try:
                data = self.data_queue.get(timeout=1)
                self.price_estimate, self.price_std = self.evaluate(data)
                print(f"{self.__class__.__name__} => Est: {self.price_estimate}, Std: {self.price_std}")
            except queue.Empty:
                continue

    def stop(self):
        self.stop_event.set()
