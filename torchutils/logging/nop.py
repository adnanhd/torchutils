from torchutils.trainer.utils import HandlerArguments, TrainerStatus
from torchutils.logging import TrainerLogger
from .utils import LoggingEvent
from typing import Dict


class NoneLogger(TrainerLogger):
    def __init__(self):
        super(NoneLogger, self).__init__()

    def open(self, args: HandlerArguments):
        pass

    @classmethod
    def getLogger(cls, event: LoggingEvent):
        return NoneLogger()

    def log_scores(self,
                   scores: Dict[str, float],
                   status: TrainerStatus):
        pass

    def update(self, n, status: TrainerStatus):
        pass

    def close(self, status: TrainerStatus):
        pass
