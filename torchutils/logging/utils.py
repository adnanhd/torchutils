import enum
from typing import NewType, Dict, Iterable, List, Union
import torch.nn as nn
from torchutils.models.utils import TrainerModel
DataFrame = NewType('DataFrame', Dict[str, Iterable[List[float]]])
Image = NewType('Image', Iterable[Iterable[Iterable[float]]])
Module = NewType('Module', Union[nn.Module, TrainerModel])


class LoggerMethodNotImplError(Exception):
    pass


class LoggerMethods(enum.Enum):
    LOG_SCORES = 0
    LOG_HPARAMS = 1
    LOG_TABLE = 2
    LOG_IMAGE = 3
    LOG_MODULE = 4


class LoggingEvent(enum.Enum):
    TRAINING_BATCH = 0
    TRAINING_EPOCH = 1
    VALIDATION_RUN = 2
    EVALUATION_RUN = 3

    @classmethod
    def getAllEvents(cls) -> Iterable[enum.Enum]:
        return [cls.TRAINING_BATCH, cls.TRAINING_EPOCH,
                cls.VALIDATION_RUN, cls.EVALUATION_RUN]
