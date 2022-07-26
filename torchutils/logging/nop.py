from torchutils.trainer.utils import IterationArguments
from torchutils.trainer.utils import IterationStatus
from .base import ScoreLogger
from typing import Dict


class NoneLogger(ScoreLogger):
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
