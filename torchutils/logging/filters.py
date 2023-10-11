import logging


def scoreFilter(record):
    return record.levelname in ("TRAIN_EPOCH", "TRAIN_STEP", "VALID_RUN",
                                "VALID_STEP", "EVAL_RUN", "EVAL_STEP")


def scoreFilterRun(record):
    return record.levelname in ("TRAIN_EPOCH", "EVAL_RUN", "VALID_RUN")


def scoreFilterStep(record):
    return record.levelname in ("VALID_STEP", "TRAIN_STEP", "EVAL_STEP")


class scoreTrainStepFilter(logging.Filter):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.counter = 0

    def filter(self, record):
        if record.levelname != 'TRAIN_STEP':
            return False
        self.counter = (self.counter + 1) % self.n
        return self.counter == 0
