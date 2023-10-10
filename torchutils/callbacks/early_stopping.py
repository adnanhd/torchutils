# Copyright © 2022 Adnan Harun Dogan
import numpy as np
import typing
from .base import TrainerCallback, StopTrainingError
from torchutils.trainer.utils import IterationInterface


class EarlyStopping(TrainerCallback):
    """
    Early stops the training if validation loss doesn't improve after a given
    patience.
    """

    def __init__(self,
                 monitor: str = 'val_loss',
                 patience: int = 7,
                 verbose: bool = False,
                 delta: float = 0.0,
                 trace_func: typing.Callable[[str], None] = print,
                 ):
        """
        Args:
            patience (int): How long to wait after last time validation loss
                            improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss
                            improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify
                           as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss = np.Inf
        self.delta = delta
        self.trace_func: typing.Callable[
            [str], None
        ] = self.logger.info if trace_func is None else trace_func
        self._check_flag_ = None

    def on_training_begin(self, hparams):
        self._check_flag_ = True

    def on_training_end(self, stat):
        if self._check_flag_:
            self.logger.warn(f'{self.__class__.__name__}.on_validation_run_end'
                             'method never called while training.')

    def on_validation_run_end(self, epoch: IterationInterface):
        score = - epoch.get_current_scores(self.monitor)[self.monitor]
        self._check_flag_ = False

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score * (1 - self.delta):
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"Stopping counter: {self.counter} out of {self.patience}")
                self.trace_func(
                    f"Best value: {self.best_score} Epoch-end value: {score}")
            if self.counter >= self.patience:
                self.early_stop = True
                raise StopTrainingError('EarlyStop stopped the model')
        else:
            self.best_score = score
            self.counter = 0
