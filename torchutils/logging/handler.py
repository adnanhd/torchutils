from typing import overload
from ..callbacks import TrainerCallback
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
        raise NotImplementedError()

    @overload
    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError()

    @overload
    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError()

    @overload
    @abstractmethod
    def close(self, *args, **kwargs):
        raise NotImplementedError()
