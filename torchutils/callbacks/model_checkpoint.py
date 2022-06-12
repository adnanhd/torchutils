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
        self.best: float = float("-inf") if maximize_score else float("inf")

    def _is_value_better(self, metric_value):
        # if aim is to maximize score, then check whether metric_value is
        # greater then our best score with a margin/confidency of delta
        # otherwise, then check whether metric_value is lower then our best
        # result with a margin/confidency of delta
        return self.maximize and metric_value > (1 + self.delta) * self.best \
            or metric_value < (1 - self.delta) * self.best

    def on_initialization(self, args: HandlerArguments):
        if self.model is None:
            self.model = args.model

    def on_training_epoch_end(self,
                              epoch: CurrentIterationStatus):
        metric_value = epoch.get_latest_scores(self.monitor)
        metric_value = metric_value[self.monitor]

        if self._is_value_better(metric_value):
            self.best = metric_value
            self.best_weights = self.model.model.state_dict()
            if self.verbose:
                self.trace_func("model weight is saved into...")
        elif self.load_back is not None \
                and self.best_weights is not None:
            self.model.model.load_state_dict(self.best_weights)
            if self.verbose:
                self.trace_func("model weight is loaded back...")

    def on_training_end(self, stat: TrainerStatus):
        if self.best_weights is not None:
            self.model.model.load_state_dict(self.best_weights)
            try:
                data = torch.load(self.save_path,
                                  # FileNotFoundError
                                  map_location=self.model.device)
                metric_value = data[self.monitor]  # KeyError
                save_model = self._is_value_better(metric_value)
            except (FileNotFoundError, KeyError):
                save_model = True

            if save_model:
                save_kwargs = {self.monitor: self.best}
                self.model.save_into_checkpoint(self.save_path,
                                                **save_kwargs)

    def on_training_begin(self, stat: TrainerStatus):
        if self.train_best and os.path.isfile(self.save_path):
            self.model.load_from_checkpoint(path=self.save_path)

    def on_evaluation_begin(self, stat: TrainerStatus):
        if self.eval_best and os.path.isfile(self.save_path):
            self.model.load_from_checkpoint(path=self.save_path)
