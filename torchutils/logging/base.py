import abc
import typing
import logging
import argparse
from ..trainer.utils import IterationArguments, IterationStatus
from .utils import DataFrame, Image, Module, LoggerMethodNotImplError


class ExperimentLogger(abc.ABC):
    __slots__ = ['config', 'handler']

    def __init__(self, handler):
        self.handler: logging.Handler = handler


class ScoreLogger(abc.ABC):
    @abc.abstractmethod
    def open(self, args: IterationArguments):
        raise LoggerMethodNotImplError

    @abc.abstractmethod
    def update(self, n, status: IterationStatus):
        raise LoggerMethodNotImplError

    @ abc.abstractmethod
    def close(self, status: IterationStatus):
        raise LoggerMethodNotImplError

    @ abc.abstractmethod
    def log_scores(self,
                   scores: typing.Dict[str, float],
                   status: IterationStatus):
        raise LoggerMethodNotImplError

    def log_hparams(self,
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

    def log_module(self,
                   module: Module,
                   status: IterationStatus, **kwargs):
        raise LoggerMethodNotImplError
