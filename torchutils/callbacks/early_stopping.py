# Copyright © 2022 Adnan Harun Dogan
import logging
import math
from .callback import TrainerCallback, StopTrainingException
from ..metrics import MetricHandler


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
                 verbose: bool = True,
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
        super().__init__(verbose=verbose)
        assert goal in ('minimize', 'maximize')

        self.monitor: str = monitor
        self.patience: int = patience
        self.delta: float = 1 + delta
        self.minimize: bool = goal == 'minimize'

        self.counter: int = 0
        self.score: float = -math.pow(-1, self.minimize) * math.inf

    def on_training_begin(self, hparams):
        print(hparams)
        if hparams['num_epochs_per_validation'] == 0:
            self.logger.warn("EarlyStopping never called while training.")

    def on_validation_run_end(self):
        score = math.pow(-1, self.minimize) * MetricHandler.get_score_average(self.monitor)

        self.logger.debug(f"Best score: {self.score} Current score: {score}")

        if self.score < score * self.delta:
            self.score = score
            self.counter = 0
            self.logger.info(f"Best Score set to {score}")
        elif self.counter < self.patience:
            self.counter += 1
            self.logger.debug(f"Plateau: {self.counter} out of {self.patience}")
        else:
            print("EARLY STOPP")
            self.logger.info("Early Stopping...")
            raise StopTrainingException
