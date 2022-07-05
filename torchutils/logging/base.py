import abc
import typing
import argparse
import collections
from .utils import DataFrame, Image, Module
from .utils import LoggerMethodNotImplError, LoggingEvent
from torchutils.trainer.utils import IterationArguments, IterationStatus


class TrainerLogger(abc.ABC):
    """
    A callback which visualises the state of each training
    and evaluation epoch using a progress bar
    """
    @classmethod
    def getLoggerGroup(cls, **kwargs
                       ) -> typing.Dict["TrainerLogger",
                                        typing.List[LoggingEvent]]:
        result = collections.defaultdict(list)
        for event in LoggingEvent.getAllEvents():
            logger = cls.getLogger(event=event, **kwargs)
            if isinstance(logger, TrainerLogger):
                result[logger].append(event)
        return dict(result)

    @classmethod
    def setEvent(
            cls,
            *events: LoggingEvent
        ) -> typing.Dict["TrainerLogger",
                         typing.List[LoggingEvent]]:
        return {cls: list(events)}

    @abc.abstractclassmethod
    def getLogger(cls, event: LoggingEvent,
                  **kwargs) -> "TrainerLogger":
        raise LoggerMethodNotImplError

    @abc.abstractmethod
    def open(self, args: IterationArguments):
        raise LoggerMethodNotImplError

    @ abc.abstractmethod
    def log_scores(self,
                   scores: typing.Dict[str, float],
                   status: IterationStatus):
        raise LoggerMethodNotImplError

    def log_hyperparams(self,
                        params: argparse.Namespace,
                        status: IterationStatus):
        raise LoggerMethodNotImplError

    def log_table(self,
                  tables: typing.Dict[str, DataFrame],
                  status: IterationStatus):
        raise LoggerMethodNotImplError

    def log_image(self,
                  images: typing.Dict[str, Image],
                  status: IterationStatus):
        raise LoggerMethodNotImplError

    def log_string(self,
                   string: str,
                   status: IterationStatus):
        raise LoggerMethodNotImplError

    @abc.abstractmethod
    def update(self, n, status: IterationStatus):
        raise LoggerMethodNotImplError

    def watch(self,
              module: Module,
              status: IterationStatus, **kwargs):
        raise LoggerMethodNotImplError

    @ abc.abstractmethod
    def close(self, status: IterationStatus):
        raise LoggerMethodNotImplError
