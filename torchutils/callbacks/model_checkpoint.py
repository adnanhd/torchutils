# Copyright © 2021 Chris Hughes
import inspect
import logging
import sys, os
import time
from abc import ABC

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from .base import TrainerCallback


class ModelCheckpoint(TrainerCallback):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, 
            monitor="val_loss", 
            save_path='best_model.ckpt', 
            verbose=False, 
            this_max=False, 
            load_back=False, 
            save_best_only=False):
        self.monitor = monitor
        self.best_weights = None
        self.verbose = verbose
        self.max = this_max
        self.save_path = save_path
        self.load_back = load_back
        self.save_best = save_best_only
        self.best = float("-inf") if this_max else float("inf")

    def _is_value_better(self, metric_value):
        return self.max and metric_value > self.best or metric_value < self.best

    def on_training_valid_end(self, trainer, **kwargs):
        metric_value = kwargs[self.monitor]

        if self._is_value_better(metric_value):
            self.best = metric_value
            self.best_weights = trainer.model.state_dict()
            if self.verbose:
                print("model weight is saved into...")
        elif self.load_back and self.best_weights is not None:
            trainer.model.load_state_dict(self.best_weights)
            if self.verbose:
                print("model weight is loaded back...")

    def on_training_end(self, trainer, **kwargs):
        if self.best_weights is not None:
            trainer.model.load_state_dict(self.best_weights)
            try:
                data = torch.load(self.save_path, 
                        map_location=trainer.device) # FileNotFoundError
                metric_value = data[self.monitor] # KeyError
                save_model = self._is_value_better(metric_value)
            except (FileNotFoundError, KeyError):
                save_model = True

            if save_model:
                save_kwargs = {self.monitor: self.best}
                trainer.save_checkpoint(path=self.save_path, **save_kwargs)

    def on_evaluation_begin(self, trainer, **kwargs):
        if os.path.isfile(self.save_path):
            trainer.load_checkpoint(path=self.save_path)

