import logging


def scoreFilter(record):
    return record.levelname in ("TRAIN_EPOCH", "TRAIN_STEP", "VALID_RUN",
                                "VALID_STEP", "EVAL_RUN", "EVAL_STEP")


class scoreFilterRun(logging.Filter):
    def __init__(self, training_epoch_per_filtering=0, validation_run_per_filtering=0):
        assert training_epoch_per_filtering >= 0
        assert validation_run_per_filtering >= 0
        super().__init__()

        self.train_threshold = training_epoch_per_filtering
        self.valid_threshold = validation_run_per_filtering
        self.train_cnt = 0
        self.valid_cnt = 0

    def filter(self, record):
        if record.levelname == 'TRAIN_EPOCH':
            if self.train_threshold == 0:
                return True
            else:
                self.train_cnt = (self.train_cnt + 1) % self.train_threshold
                return self.train_cnt == 0
        if record.levelname == 'VALID_RUN':
            if self.valid_threshold == 0:
                return True
            else:
                self.valid_cnt = (self.valid_cnt + 1) % self.valid_threshold
                return self.valid_cnt == 0
        return True


class scoreFilterStep(logging.Filter):
    def __init__(self, training_step_per_filtering=0, validation_step_per_filtering=0):
        assert training_step_per_filtering >= 0
        assert validation_step_per_filtering >= 0
        super().__init__()

        self.train_threshold = training_step_per_filtering
        self.valid_threshold = validation_step_per_filtering
        self.train_cnt = 0
        self.valid_cnt = 0

    def filter(self, record):
        if record.levelname == 'TRAIN_STEP':
            if self.train_threshold == 0:
                return True
            else:
                self.train_cnt = (self.train_cnt + 1) % self.train_threshold
                return self.train_cnt == 0
        if record.levelname == 'VALID_STEP':
            if self.valid_threshold == 0:
                return True
            else:
                self.valid_cnt = (self.valid_cnt + 1) % self.valid_threshold
                return self.valid_cnt == 0
        return True
