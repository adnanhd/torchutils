from typing import List, Any
import warnings
from .base import TrainerLogger
from .utils import LoggerMethodNotImplError, LoggerMethods
from ..trainer.status import IterationStatus
from ..trainer.arguments import Hyperparameter


class LoggerHandler(object):
    __slots__ = ['__loggers__']

    @classmethod
    def __get_validators__(cls):
        yield cls.validator

    @classmethod
    def validator(cls, other):
        if isinstance(other, LoggerHandler):
            return other
        else:
            raise ValueError

    def __init__(self):
        self.__loggers__: List[TrainerLogger] = list()

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

    def log(self, method: LoggerMethods, message: Any, status: IterationStatus):
        any_overwritten_method = False
        for logger in self.__loggers__:
            try:
                log_fn = getattr(logger, method.name.lower())
                log_fn(message, status=status)
            except LoggerMethodNotImplError:
                continue
            else:
                any_overwritten_method = True
        if not any_overwritten_method:
            warnings.warn(
                f"No logger has an overriden method for {method.name.lower()}"
            )

    def initialize_loggers(self, hparams: Hyperparameter):
        for logger in self.__loggers__:
            logger.open(hparams)

    def finalize_loggers(self, status: IterationStatus):
        for logger in self.__loggers__:
            logger.close(status)

    def update_loggers(self, status: IterationStatus, n: int = 1):
        for logger in self.__loggers__:
            logger.update(n=n, status=status)
