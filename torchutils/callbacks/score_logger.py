from .callback import TrainerCallback
import pydantic
import typing
from .utils import LogLevelEnum


class AverageScoreLogger(TrainerCallback):
    def __init__(self, *score_names):
        super().__init__(readable_scores=set(score_names))

    def on_training_epoch_begin(self, epoch_index: int):
        self._buffer['epoch_index'] = epoch_index

    def on_training_step_begin(self, batch_index: int):
        self._buffer['batch_index'] = batch_index

    def on_training_step_end(self, batch_index, batch, batch_output):
        self.log(level=LogLevelEnum.TRAINING_STEP_END.value, msg=self.get_score_value().union(self._buffer))
    
    def on_validation_step_end(self):
        self.log(level=LogLevelEnum.VALIDATION_STEP_END.value, msg=self.get_score_value().union(self._buffer))

    def on_training_epoch_end(self):
        self.log(level=LogLevelEnum.TRAINING_EPOCH_END.value, msg=self.get_score_averages().union(self._buffer))