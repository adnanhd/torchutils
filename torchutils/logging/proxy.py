import warnings
from typing import List, Dict, Optional, Any
from enum import Enum
import argparse
from abc import ABC
from .base import LoggerMethodNotImplError, TrainerLogger


class LoggingEvent(Enum):
    TRAINING_BATCH = 0
    TRAINING_EPOCH = 1
    VALIDATION_RUN = 2
    EVALUATION_RUN = 3


class LoggingLevel(Enum):
    SCALAR = 'log_scalar'
    HPARAM = 'log_hyperparams'
    TABLE = 'log_table'
    IMAGE = 'log_image'
    STRING = 'log_string'


def __foreach_logger__(mthd):
    def wrapper_method(self,
                       msg: Any,
                       event: Optional[LoggingEvent] = None,
                       **kwargs):
        assert isinstance(event, LoggingEvent) or event is None
        # assert isinstance(level, LoggingLevel)
        if event is None:
            event = self._event
        if self._logger_dict_.__len__() != 0:
            any_overwritten_method = False
            for logger in self.__loggers__[event]:
                try:
                    getattr(logger, mthd.__name__)(msg, **kwargs)
                except LoggerMethodNotImplError:
                    continue
                else:
                    any_overwritten_method = True
            if not any_overwritten_method:
                warnings.warn(
                    f"No logger has an overwritten method {mthd.__name__}"
                )
        else:
            warnings.warn(
                f"No logger added for event {event}"
            )
    return wrapper_method


class LoggerProxy(ABC):

    __slots__ = ['__loggers__', '_event']

    def __init__(self, loggers: Dict[str, TrainerLogger]):
        self.__loggers__ = loggers
        self._event: Optional[LoggingEvent] = None

    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, event):
        if isinstance(event, LoggingEvent):
            self._event = event
        else:
            raise AssertionError(
                f"event must be of LoggingEvent, not {type(event)}"
            )

    @__foreach_logger__
    def log_scores(
            self, scores: Dict[str, float],
            step: Optional[int] = None
    ):
        pass

    @__foreach_logger__
    def log_hyperparams(
            self, params: argparse.Namespace, *args, **kwargs
    ):
        pass

    @__foreach_logger__
    def log_table(
            self, key: str, table: Dict[str, List[Any]]
    ):
        pass

    @__foreach_logger__
    def log_image(
            self, key: str, images: List[Any],
            step: Optional[int] = None
    ):
        pass

    @__foreach_logger__
    def log_string(self, msg):
        pass

    @__foreach_logger__
    def log_module(self, module):
        pass

    @__foreach_logger__
    def update(self, n=1):
        pass
