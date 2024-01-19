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
    _epoch: int = pydantic.PrivateAttr(0)
    _nepv: int = pydantic.PrivateAttr()

    @pydantic.field_validator('delta')
    def validate_delta(cls, value, values: pydantic.ValidationInfo):
        return (1 + value) if values.data['goal'].value == 'maximize' else (1 - value)

    @property
    def maximize(self):
        return self.goal.value == 'maximize'
    

    def __init__(self, monitor: str, verbose=False, **kwds):
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
        if isinstance(verbose, int):
            level = verbose
        else:
            level = 10 if verbose else 20
        super().__init__(readable_scores={monitor}, monitor=monitor, level=level, **kwds)

    def on_training_begin(self, hparams):
        self._nepv = hparams['num_epochs_per_validation']
        if hparams['num_epochs_per_validation'] == 0:
            self.log_warn("EarlyStopping will not be called while training.")

    def on_validation_run_begin(self, epoch_index: int):
        self._epoch = epoch_index

    def on_validation_run_end(self):
        # score <- -monitored_score if self.maximize else monitored_score
        # score <- monitored_score if maximize else -monitored_score
        score = self.get_score_averages()[self.monitor]
        self.log_debug(f'Best score: {self._score} Current Score: {score}')
        _score = score if self.maximize else -score

        if self._score * self.delta < _score:
            self._score = _score
            self._counter = 0
            self.log_info(f"Best {self.monitor} score set to {score}")
        elif self._counter < self.patience:
            self._counter += self._nepv
            self.log_info(f"Plateau: {self._counter} out of {self.patience}")
        else:
            delta = (1 - self.delta) if self.maximize else (1 - self.delta)
            self.log_warn(f"Early Stopping @ epoch {self._epoch} as {self.monitor}={score:.3e} "
                          f"not optimized {delta * 100:.2f}% w/in the last {self.patience} calls...")
            raise StopTrainingException
