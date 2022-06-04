from torchutils.utils.pydantic import HandlerArguments
from torchutils.logging import TrainerLogger


class NoneLogger(TrainerLogger):
    def __init__(self):
        super(NoneLogger, self).__init__()

    def open(self, args: HandlerArguments):
        pass

    def log_score(self, **kwargs):
        pass

    def log_info(self, msg):
        pass
    
    def log_error(self, msg):
        pass

    def _flush_step(self):
        pass

    def _flush_epoch(self):
        pass

    def close(self):
        pass

