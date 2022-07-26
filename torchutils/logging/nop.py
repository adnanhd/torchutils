from typing import Dict
from .base import TrainerLogger
from ..trainer.status import IterationStatus
from ..trainer.arguments import IterationArguments


class NoneLogger(TrainerLogger):
    def __init__(self):
        super(NoneLogger, self).__init__()

    def open(self, args: IterationArguments):
        pass

    def log_scores(self,
                   scores: Dict[str, float],
                   status: IterationStatus):
        pass

    def update(self, n, status: IterationStatus):
        pass

    def close(self, status: IterationStatus):
        pass
