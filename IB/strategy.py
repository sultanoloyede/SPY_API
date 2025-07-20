from abc import ABC, abstractmethod
from threading import Thread, Event
from typing import Tuple
import queue

class Strategy(ABC, Thread):
    def __init__(self):
        """
        Abstract defintion for a strategy observer class.
        Sets up a thread-safe queue for data communication and an event to signal stopping the strategy.
        """

        super().__init__()
        self.data_queue = queue.Queue()  # Thread-safe queue
        self.stop_event = Event()

    def on_new_data(self, data):
        self.data_queue.put(data)

    @abstractmethod
    def evaluate(self, data) -> Tuple[float, float]:
        pass

    def run(self):
        while not self.stop_event.is_set():
            try:
                data = self.data_queue.get(timeout=1)
                estimation, std = self.evaluate(data)
                print(f"{self.__class__.__name__} => Est: {estimation}, Std: {std}")
            except queue.Empty:
                continue

    def stop(self):
        self.stop_event.set()
