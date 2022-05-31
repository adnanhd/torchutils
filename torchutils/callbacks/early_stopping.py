# Copyright © 2022 Adnan Harun Dogan
from .base import TrainerCallback, StopTrainingError
import gym

class Environment(gym.Env):
    pass


class EarlyStopping(TrainerCallback):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, monitor='val_loss', patience=7, verbose=False, delta=0, trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def on_training_valid_end(self, trainer, epoch=None, **kwargs):
        score = -kwargs[self.monitor]

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"BOK YE AMK EarlyStopping counter: {self.counter} out of {self.patience}"
                    f"Best value: {self.best} Epoch-end value: {score}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                raise StopTrainingError(f'EarlyStop stopped the model')
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop

