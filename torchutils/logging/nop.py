from torchutils.trainer.utils import IterationArguments, IterationStatus
from torchutils.logging import TrainerLogger
from .utils import LoggingEvent
from typing import Dict


class NoneLogger(TrainerLogger):
    def __init__(self):
        super(NoneLogger, self).__init__()

    def open(self, args: IterationArguments):
        pass

    @classmethod
    def getLogger(cls, event: LoggingEvent):
        return NoneLogger()

    def log_scores(self,
                   scores: Dict[str, float],
                   status: IterationStatus):
        pass

    def update(self, n, status: IterationStatus):
        pass

    def close(self, status: IterationStatus):
        pass
