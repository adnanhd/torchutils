# Copyright © 2022 Adnan Harun Dogan
from .callback import TrainerCallback, StopTrainingException
from .utils import GoalEnum
import pydantic


class EarlyStopping(TrainerCallback):
    """
    Early stops the training if validation loss doesn't improve after a given
    patience.
    """
    monitor: str
    goal: GoalEnum
    patience: int = 7
    delta: float = pydantic.Field(0.0, default_validate=True)

    _counter: int = pydantic.PrivateAttr(0)
    _score: float = pydantic.PrivateAttr(-float('inf'))

    @pydantic.field_validator('delta')
    def validate_delta(cls, value, values: pydantic.ValidationInfo):
        return (1 + value) if values.data['goal'].value == 'maximize' else (1 - value)

    @property
    def maximize(self):
        return self.goal.value == 'maximize'
    

    def __init__(self,
                 monitor: str,
                 **kwds
                 ):
        """
        Args:
            monitor (str): The metric or quantity to be monitored.
                            Default: val_loss
            maximize (bool): How improving the monitored qunatity defined.
                            If true, the maximum is better, otherwise,
                            vice versa.
                            Default: True
            patience (int): How long to wait after last time the monitored
                            quantity has improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify
                           as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each improvement in
                            the monitored quantity
                            Default: False
        """
        super().__init__(readable_scores={monitor}, monitor=monitor, **kwds)

    def on_training_begin(self, hparams):
        if hparams['num_epochs_per_validation'] == 0:
            self.log_warn("EarlyStopping will not be called while training.")

    def on_validation_run_end(self):
        # score <- -monitored_score if self.maximize else monitored_score
        # score <- monitored_score if maximize else -monitored_score
        score = self.get_score_averages()[self.monitor]
        self.log_debug(f'Best score: {self._score} Current Score: {score}')
        _score = score if self.maximize else -score

        if self._score * self.delta < _score:
            self._score = _score
            self._counter = 0
            self.log_info(f"Best Score set to {score}")
        elif self._counter < self.patience:
            self._counter += 1
            self.log_debug(f"Plateau: {self._counter} out of {self.patience}")
        else:
            self.log_info("Early Stopping...")
            raise StopTrainingException
