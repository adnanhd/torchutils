from typing import overload
from abc import ABC, abstractmethod


class TrainerLogger(ABC):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """
    @abstractmethod
    def open(self):
        ...

    @abstractmethod
    def log(self, **kwargs):
        ...

    @abstractmethod
    def _flush(self):
        ...

    @abstractmethod
    def close(self):
        ...


class LoggingHandler(ABC):
    def init(self):
        ...

    def log(self):
        ...

    def step(self):
        ...

    def close(self):
        ...





