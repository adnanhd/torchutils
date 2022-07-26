import abc
import enum
import argparse
import warnings
from .base import TrainerLogger
from ..trainer.utils import IterationStatus
from typing import Dict, List, Any
from .utils import LoggerMethodNotImplError, LoggingEvent, Image, DataFrame


class ProxyMethods(enum.Enum):
    LOG_SCORES = 0
    LOG_HYPERPARAMS = 1
    LOG_TABLE = 2
    LOG_IMAGE = 3
    LOG_STRING = 4
    UPDATE = 5


class LoggerProxy(abc.ABC):

    __slots__ = ['__loggers__', '__event__', '__status__']

    def __init__(self,
                 loggers: Dict[str, TrainerLogger],
                 _event_ptr: List[LoggingEvent],
                 _status_ptr: List[IterationStatus]):
        assert all(
            all(isinstance(logger, TrainerLogger) for logger in logger_list)
            for logger_list in loggers.values()), (
            f"{loggers.values()} must be a list of TrainerLogger"
        )
        # assert len(_status_ptr) == 1 and isinstance(
        #     _status_ptr[0], IterationStatus), f"{_status_ptr} is not IterationStatus"
        # assert len(_event_ptr) == 1 and isinstance(
        #     _event_ptr[0], LoggingEvent), f"{_event_ptr} is not LoggingEvent"
        self.__loggers__ = loggers
        self.__event__: List[LoggingEvent] = _event_ptr
        self.__status__: List[IterationStatus] = _status_ptr

    @property
    def event(self) -> LoggingEvent:
        return self.__event__[0]

    def __log__(self, method: ProxyMethods, message: Any):
        if self.__loggers__[self.event].__len__() != 0:
            any_overwritten_method = False
            for logger in self.__loggers__[self.event]:
                try:
                    log_fn = getattr(logger, method.name.lower())
                    log_fn(message, status=self.__status__[0])
                except LoggerMethodNotImplError:
                    continue
                else:
                    any_overwritten_method = True
            if not any_overwritten_method:
                warnings.warn(
                    f"No logger has an overriden method {method.name.lower()}"
                )
        else:
            warnings.warn(
                f"No logger added for {self.event.name}"
            )

    def log_table(self, tables: Dict[str, DataFrame]):
        self.__log__(ProxyMethods.LOG_TABLE, message=tables)

    def log_image(self, images: Dict[str, Image]):
        self.__log__(ProxyMethods.LOG_IMAGE, message=images)

    def log_string(self, string: str):
        self.__log__(ProxyMethods.LOG_STRING, message=string)

    def log_scores(self, scores: Dict[str, float]):
        self.__log__(ProxyMethods.LOG_SCORES, message=scores)

    def log_hyperparams(self, params: argparse.Namespace):
        self.__log__(ProxyMethods.LOG_HYPERPARAMS, message=params)

    def update(self, n: int = 1):
        self.__log__(ProxyMethods.UPDATE, message=n)
