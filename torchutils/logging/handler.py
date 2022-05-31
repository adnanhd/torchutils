from typing import overload
from abc import ABC, abstractmethod


class TrainerLogger(ABC):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """

    def __init__(self):
        super(TrainerLogger, self).__init__()

    @overload
    @abstractmethod
    def open(self, *args, **kwargs):
        ...

    @overload
    @abstractmethod
    def log(self, *args, **kwargs):
        ...

    @overload
    @abstractmethod
    def update(self, *args, **kwargs):
        ...

    @overload
    @abstractmethod
    def close(self, *args, **kwargs):
        ...

