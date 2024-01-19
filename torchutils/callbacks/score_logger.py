from .callback import TrainerCallback
import pydantic
import typing
from .utils import LogLevelEnum


class AverageScoreLogger(TrainerCallback):
    def __init__(self, *score_names, level=10):
        super().__init__(readable_scores=set(score_names))
        self._logger.setLevel(level)

    ####################################################################

    def on_training_epoch_begin(self, epoch_index: int):
        self._buffer['epoch_index'] = epoch_index

    def on_training_step_begin(self, batch_index: int):
        self._buffer['batch_index'] = batch_index

    def on_training_step_end(self, batch_index, batch, batch_output):
        values = self.get_score_values()
        values.update(self._buffer)
        self.log(level=LogLevelEnum.TRAINING_STEP_END.value, msg=values)

    def on_training_epoch_end(self):
        averages = self.get_score_averages()
        averages['epoch_index'] = self._buffer['epoch_index']
        self.log(level=LogLevelEnum.TRAINING_EPOCH_END.value, msg=averages)

    ####################################################################

    def on_validation_step_begin(self, batch_index: int):
        self._buffer['batch_index'] = batch_index
    
    def on_validation_step_end(self, batch_index, batch, batch_output):
        values = self.get_score_values()
        values.update(self._buffer)
        self.log(level=LogLevelEnum.VALIDATION_STEP_END.value, msg=values)

    def on_validation_run_end(self):
        averages = self.get_score_averages()
        averages['epoch_index'] = self._buffer['epoch_index']
        self.log(level=LogLevelEnum.VALIDATION_END.value, msg=averages)

    ####################################################################

    def on_evaluation_step_begin(self, batch_index: int):
        self._buffer['batch_index'] = batch_index

    def on_evaluation_step_end(self, batch_index, batch, batch_output):
        values = self.get_score_values()
        values['batch_index'] = self._buffer['batch_index']
        self.log(level=LogLevelEnum.EVALUATION_STEP_END.value, msg=values)

    def on_evaluation_run_end(self):
        self.log(level=LogLevelEnum.EVALUATION_END.value, msg=self.get_score_averages())