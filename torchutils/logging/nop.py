from torchutils.utils.pydantic import HandlerArguments
from torchutils.logging import TrainerLogger
from typing import Dict, Optional


class NoneLogger(TrainerLogger):
    def __init__(self):
        super(NoneLogger, self).__init__()

    def open(self, args: HandlerArguments):
        pass

    def log_scores(self,
                   scores: Dict[str, float],
                   step: Optional[int] = None):
        pass

    def close(self):
        pass
