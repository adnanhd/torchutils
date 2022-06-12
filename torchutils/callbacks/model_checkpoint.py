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
                 save_path='best_model.ckpt',
                 trainer_model: TrainerModel = None,
                 trace_func: Callable[str, None] = print,
                 verbose: bool = False,
                 this_max: bool = False,
                 load_back: int = None,
                 save_best_only: bool = False):
        self.monitor: str = monitor
        self.best_weights = None
        self.verbose: bool = verbose
        self.max: bool = this_max
        self.save_path: str = save_path
        self.load_back: int = load_back
        self.trace_func: Callable[str, None] = trace_func
        self.save_best: bool = save_best_only
        self.model: TrainerModel = trainer_model
        self.best: float = float("-inf") if this_max else float("inf")

    def _is_value_better(self, metric_value):
        if self.max and metric_value > self.best:
            return True
        if metric_value < self.best:
            return True
        return False

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
                self.model.save_checkpoint(self.save_path,
                                           **save_kwargs)

    def on_training_begin(self, stat: TrainerStatus):
        if os.path.isfile(self.save_path):
            self.model.load_checkpoint(path=self.save_path)

    def on_evaluation_begin(self, stat: TrainerStatus):
        if os.path.isfile(self.save_path):
            self.model.load_checkpoint(path=self.save_path)
