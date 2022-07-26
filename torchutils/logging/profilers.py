import typing
import warnings
from abc import ABC
from .base import ExperimentLogger
import logging


class ExperimentProfiler(ABC):
    __slots__ = ['__loggers__']

    def __init__(self):
        self.__loggers__: typing.List[ExperimentLogger] = list()

    @property
    def handlers(self) -> typing.List[logging.Handler]:
        return [logger.handler for logger in self.__loggers__]

    def add_loggers(self, loggers: typing.List[ExperimentLogger]):
        for logger in loggers:
            self.add_logger(logger)

    def add_logger(self, logger: ExperimentLogger):
        assert isinstance(logger, ExperimentLogger)
        self.__loggers__.append(logger)

    def remove_loggers(self, loggers: typing.List[ExperimentLogger]):
        for logger in loggers:
            self.remove_logger(logger)

    def remove_logger(self, logger: ExperimentLogger):
        assert isinstance(logger, ExperimentLogger)
        if logger in self.__loggers__:
            self.__loggers__.remove(logger)
        else:
            warnings.warn(f"{logger} is not in loggers", UserWarning)

    def clear_loggers(self):
        self.__loggers__.clear()
