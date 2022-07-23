from typing import List
import warnings
from abc import ABC
from .base import TrainerLogger


class LoggerHandler(ABC):
    __slots__ = ['__loggers__']

    def __init__(self):
        self.__loggers__ = list()

    def add_loggers(self, loggers: List[TrainerLogger]):
        for logger in loggers:
            self.add_logger(logger)

    def add_logger(self, logger: TrainerLogger):
        assert isinstance(logger, TrainerLogger)
        self.__loggers__.append(logger)

    def remove_loggers(self, loggers: List[TrainerLogger]):
        for logger in loggers:
            self.remove_logger(logger)

    def remove_logger(self, logger: TrainerLogger):
        assert isinstance(logger, TrainerLogger)
        if logger in self.__loggers__:
            self.__loggers__.remove(logger)
        else:
            warnings.warn(f"{logger} is not in loggers", UserWarning)

    def clear_loggers(self):
        self.__loggers__.clear()
