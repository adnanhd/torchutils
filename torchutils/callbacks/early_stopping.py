# Copyright © 2022 Adnan Harun Dogan
import logging
import math
from .callback import TrainerCallback, StopTrainingException


class EarlyStopping(TrainerCallback):
    """
    Early stops the training if validation loss doesn't improve after a given
    patience.
    """

    def __init__(self,
                 monitor: str,
                 goal: str = 'minimize',
                 patience: int = 7,
                 delta: float = 0.0,
                 verbose: bool = False,
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
        super().__init__(level=logging.DEBUG if verbose else logging.INFO)
        assert goal in ('minimize', 'maximize')

        self.monitor: str = monitor
        self.patience: int = patience
        self.maximize: bool = goal == 'maximize'
        self.delta: float = (1 + delta) if self.maximize else (1 - delta)

        self.counter: int = 0
        self.score: float = -math.inf

    def on_training_begin(self, hparams):
        if hparams['num_epochs_per_validation'] == 0:
            self.logger.warn("EarlyStopping never called while training.")

    def on_validation_run_end(self):
        # score <- -monitored_score if self.maximize else monitored_score
        # score <- monitored_score if maximize else -monitored_score
        score = self.scores[self.monitor]
        score = score if self.maximize else -score

        self.logger.debug(f"Best score: {self.score if self.maximize else -self.score} "
                          f"Current score: {score if self.maximize else -score}")
        if self.score * self.delta < score:
            self.score = score
            self.counter = 0
            self.logger.info(f"Best Score set to {score if self.maximize else -score}")
        elif self.counter < self.patience:
            self.counter += 1
            self.logger.debug(f"Plateau: {self.counter} out of {self.patience}")
        else:
            self.logger.info("Early Stopping...")
            raise StopTrainingException
