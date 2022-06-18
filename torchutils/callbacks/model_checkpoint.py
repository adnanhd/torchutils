# Copyright © 2021 Chris Hughes
import os
import torch
from .base import TrainerCallback
from typing import Callable
from torchutils.utils.pydantic import (
    TrainerModel,
    HandlerArguments,
    TrainerStatus,
    CurrentIterationStatus
)


class ModelCheckpoint(TrainerCallback):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self,
                 monitor="loss",
                 maximize_score: bool = False,
                 delta: float = 0.0,
                 save_path='model_checkpoint.ckpt',
                 trainer_model: TrainerModel = None,
                 trace_func: Callable[[str], None] = print,
                 verbose: bool = False,
                 load_back: int = None,
                 load_before_train: bool = False,
                 load_before_evaluate: bool = True,
                 save_best_only: bool = False):
        self.monitor: str = monitor
        self.best_weights = None
        self.delta: float = delta
        self.verbose: bool = verbose
        self.maximize: bool = maximize_score
        self.save_path: str = save_path
        self.load_back: int = load_back
        self.trace_func: Callable[[str], None] = trace_func
        self.save_best: bool = save_best_only
        self.model: TrainerModel = trainer_model
        self.train_best: bool = load_before_train
        self.eval_best: bool = load_before_evaluate
        self.best_score: float = float(
            "-inf") if maximize_score else float("inf")

    def _is_value_better(self, metric_value):
        # if aim is to maximize score, then check whether metric_value is
        # greater then our best score with a margin/confidency of delta
        # otherwise, then check whether metric_value is lower then our best
        # result with a margin/confidency of delta
        return self.maximize and metric_value > (1 + self.delta) * self.best_score \
            or metric_value < (1 - self.delta) * self.best_score

    def _conditional_load_into_checkpoint(self):
        try:
            self.model.load_from_checkpoint(
                halt_condition=self._if_upcoming_worse,
                path=self.save_path
            )
        except FileNotFoundError:
            self.model.save_into_checkpoint(
                path=self.save_path,
                scores={self.monitor: self.best_score}
            )

    def _load_best_score(self, state):
        return state['scores'][self.monitor]

    def _load_from_model(self, model_score, verbose=True):
        self.best_score = model_score
        self.best_weights = self.model.model.state_dict()
        if verbose and self.verbose:
            self.trace_func(
                "the best weights are loaded back from the model..."
            )

    def _save_into_model(self, verbose=True):
        self.model.model.load_state_dict(self.best_weights)
        if verbose and self.verbose:
            self.trace_func(
                "the best weights is saved into the model..."
            )

    def _if_upcoming_worse(self, state):
        """ Halts if score in the disk is no better than that of the memory
        and if score in the disk is absent, then it does not halt. """
        try:
            return not self._is_value_better(self._load_best_score(state))
        except KeyError:
            return False

    def on_initialization(self, args: HandlerArguments):
        if self.model is None:
            self.model = args.model

    def on_training_epoch_end(
            self, epoch: CurrentIterationStatus
    ):
        metric_value = epoch.get_current_scores(self.monitor)[self.monitor]

        if self._is_value_better(metric_value):
            self._load_from_model(metric_value)
        elif self.load_back is not None:
            if self.best_weights is not None:
                self._save_into_model()

    def on_training_end(self, stat: TrainerStatus):
        if self.best_weights is not None:
            self._save_into_model()
        self._conditional_load_into_checkpoint()

    def on_training_begin(self, stat: TrainerStatus):
        if self.train_best and os.path.isfile(self.save_path):
            state = self.model.load_from_checkpoint(
                halt_condition=self._if_upcoming_worse,
                path=self.save_path
            )

            try:
                best_score = self._load_best_score(state)
            except KeyError:
                best_score: float = float(
                    "-inf") if self.maximize else float("inf")
            self._load_from_model(best_score)

    def on_evaluation_begin(self, stat: TrainerStatus):
        if self.eval_best and os.path.isfile(self.save_path):
            state = self.model.load_from_checkpoint(
                halt_condition=self._if_upcoming_worse,
                path=self.save_path
            )

            try:
                best_score = self._load_best_score(state)
            except KeyError:
                best_score: float = float(
                    "-inf") if self.maximize else float("inf")
            self._load_from_model(best_score)
