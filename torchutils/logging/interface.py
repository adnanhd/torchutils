import argparse
from typing import Dict, Sequence

from .handler import LoggerHandler
from .utils import Image, DataFrame, LoggerMethods

from ..models.utils import TrainerModel
from ..trainer.status import IterationStatus


class LoggerInterface(object):
    __slots__ = ['__handler__', '__status__']

    @classmethod
    def __get_validators__(cls):
        yield cls.validator

    @classmethod
    def validator(cls, other):
        if isinstance(other, LoggerInterface):
            return other
        else:
            raise ValueError

    def __init__(self,
                 handler: LoggerHandler,
                 status: Sequence[IterationStatus]):
        self.__status__ = status
        self.__handler__ = handler

    def log_scores(self,
                   scores: Dict[str, float]):
        self.__handler__.log(LoggerMethods.LOG_SCORES,
                             message=scores,
                             status=self.__status__[0])

    def log_hparams(self,
                    params: argparse.Namespace):
        self.__handler__.log(LoggerMethods.LOG_HPARAMS,
                             message=params,
                             status=self.__status__[0])

    def log_table(self,
                  tables: Dict[str, DataFrame]):
        self.__handler__.log(LoggerMethods.LOG_TABLE,
                             message=tables,
                             status=self.__status__[0])

    def log_image(self,
                  images: Dict[str, Image]):
        self.__handler__.log(LoggerMethods.LOG_IMAGE,
                             message=images,
                             status=self.__status__[0])

    def log_module(self,
                   module: TrainerModel):
        self.__handler__.log(LoggerMethods.LOG_MODULE,
                             message=module,
                             status=self.__status__[0])
