from abc import ABC, abstractmethod
from threading import Thread, Event
from typing import Tuple
import queue

from utils.logger import logger

class Strategy(ABC, Thread):
    """
    Abstract defintion for a strategy observer class.
    Sets up a thread-safe queue for data communication and an event to signal stopping the strategy.
    """
    def __init__(self):
        super().__init__()
        self.data_queue = queue.Queue()  # Thread-safe queue
        self.stop_event = Event()
        self.price_estimate: float = 0
        self.price_std:float = 0


    def on_new_data(self, data):
        self.data_queue.put(data)

    @abstractmethod
    def evaluate(self, data) -> None:
        pass

    def run(self):
        while not self.stop_event.is_set():
            try:
                logger.debug("Waiting for new data in the queue...")
                data = self.data_queue.get(timeout=1)
                self.evaluate(data)
                logger.info(f"{self.__class__.__name__} => Est: {self.price_estimate:.4f}, Std: {self.price_std:.4f}")
            except queue.Empty:
                logger.debug("No new data available in queue, waiting...")

    def stop(self):
        self.stop_event.set()
