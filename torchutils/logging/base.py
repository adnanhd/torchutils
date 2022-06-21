import abc
import typing
import argparse
from .utils import LoggerMethodNotImplError, LoggingEvent


class TrainerLogger(abc.ABC):
    """
    A callback which visualises the state of each training
    and evaluation epoch using a progress bar
    """
    @abc.abstractclassmethod
    def getLogger(self, event: LoggingEvent):
        raise LoggerMethodNotImplError

    @abc.abstractmethod
    def open(self):
        raise LoggerMethodNotImplError

    @abc.abstractmethod
    def log_scores(self,
                   scores: typing.Dict[str, float],
                   step: typing.Optional[int] = None):
        raise LoggerMethodNotImplError

    def log_hyperparams(self,
                        params: argparse.Namespace,
                        *args,
                        **kwargs):
        raise LoggerMethodNotImplError

    def log_table(self,
                  key: str,
                  table: typing.Dict[str, typing.List[typing.Any]]):
        raise LoggerMethodNotImplError

    def log_image(self,
                  key: str,
                  Images: typing.List[typing.Any],
                  step: typing.Optional[int] = None):
        raise LoggerMethodNotImplError

    def log_string(self, msg):
        raise LoggerMethodNotImplError

    def log_module(self, module):
        raise LoggerMethodNotImplError

    @abc.abstractmethod
    def close(self):
        raise LoggerMethodNotImplError
