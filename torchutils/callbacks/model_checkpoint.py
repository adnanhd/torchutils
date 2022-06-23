# Copyright © 2021 Chris Hughes
import os
import torch
import warnings
from .base import TrainerCallback
from typing import Callable
from torchutils.utils.pydantic import (
    TrainerModel,
    HandlerArguments,
    TrainerStatus,
    CurrentIterationStatus
)


class ModelCheckpoint(TrainerCallback):
    """
    Handles save and load operations between TrainerModel and Memory.

    Parameters
    ----------
        monitor : str
            name of the score to be kept track of
        trainer_model : TrainerModel
            the model of which weights are to be saved from and stored back
        save_path : str
            path on the memory of the TRAINER_MODEL's state_dict
        minimize_score : bool
            saves the TRAINER_MODEL's state dict when the monitored score is
            minimized with a margin of DELTA if True, or the score is maximized
            otherwise.
        delta : float
            a safety margin percentile of the current score from the saved
            score that makes the current checkpoint better than the saved one
        verbose : bool
            prints when checkpoint is saved, reset, and stored back the model
        load_back_per_epochs : int
            sets TRAINER_MODEL's state dict on the training end of every
            LOAD_BACK_PER_EPOCHS epochs. It is disabled when is zero.
        init_from_checkpoint : bool
            initializes the best checkpoint in the ModelCheckpoint with the one
            stored on SAVE_PATH on_initialization if True, or with None otwise.
        halt_into_checkpoint : bool
            stores the best checkpoint allocated in the ModelCheckpoint back to
            the SAVE_PATH on_termination if True, does noting otherwise.
        save_only_best_model : bool
            does not store the best checkpoint in the ModelCheckpoint back to
            the SAVE_PATH on_termination when the one on the same path have
            better score if True, does not check score parameter while storing
            otherwise.
        eval_with_best_model : bool
            loads the best checkpoint in the ModelCheckpoint to the TRAINER_MODEL
            on_evaluation_begin.
    """

    def __init__(self,
                 monitor="loss",
                 trainer_model: TrainerModel = None,
                 save_path='model_checkpoint.ckpt',
                 maximize_score: bool = False,
                 delta: float = 0.0,
                 # TODO: replace this param with TrainerLogger
                 trace_func: Callable[[str], None] = print,
                 verbose: bool = False,
                 load_back_per_epochs: int = 0,
                 init_from_checkpoint: bool = False,
                 halt_into_checkpoint: bool = False,
                 save_only_best_model: bool = True,
                 eval_with_best_model: bool = False,
                 ):
        self.monitor: str = monitor
        self.model: TrainerModel = trainer_model
        self.delta: float = delta
        self.verbose: bool = verbose
        self.maximize: bool = maximize_score
        self.save_path: str = save_path
        self.trace_func: Callable[[str], None] = trace_func

        self.load_back_per_epochs: int = load_back_per_epochs
        self.init_from_checkpoint: bool = init_from_checkpoint
        self.halt_into_checkpoint: bool = halt_into_checkpoint
        self.save_only_best_model: bool = save_only_best_model
        self.eval_with_best_model: bool = eval_with_best_model

        self._best_weights = None
        self._best_score: float = float(
            "-inf") if maximize_score else float("inf")

        if isinstance(self.load_back_per_epochs, int) \
                and self.load_back_per_epochs > 0:
            def on_training_epoch_begin(self, status: TrainerStatus):
                if self._best_weights is not None and \
                        status.current_epoch % self.load_back_per_epochs == 0:
                    self.set_checkpoint_into_model()
            self.on_training_epoch_begin = on_training_epoch_begin

    def is_value_better(self, metric_value):
        # if aim is to maximize score, then check whether metric_value is
        # greater then our best score with a margin/confidency of delta
        # otherwise, then check whether metric_value is lower then our best
        # result with a margin/confidency of delta
        return self.maximize and metric_value > (1 + self.delta) * self._best_score \
            or metric_value < (1 - self.delta) * self._best_score

    def save_into_checkpoint(self) -> None:
        if self.save_only_best_model and os.path.isfile(self.save_path):
            checkpoint = torch.load(self.save_path)
            try:
                best_score = checkpoint['scores'][self.monitor]
            except KeyError:
                save_model = True
            else:
                save_model = self.is_value_better(metric_value=best_score)
        else:
            save_model = True
        if save_model:
            checkpoint = {'state_dict': self._best_weights,
                          'scores': {self.monitor: self._best_score}}
            torch.save(checkpoint, self.save_path)

    def load_from_checkpoint(self):
        if os.path.isfile(self.save_path):
            checkpoint = torch.load(
                self.save_path, map_location=self.model.device)
            self._best_weights = checkpoint['state_dict']
        else:
            warnings.warn(
                "ModelCheckpoint.load_from_checkpoint: file could not "
                f"found in {self.save_path}")

    def get_checkpoint_from_model(self, model_score):
        self._best_score = model_score
        self._best_weights = self.model.state_dict()
        if self.verbose:
            self.trace_func(
                "A newer checkpoint is obtained from the model."
            )

    def set_checkpoint_into_model(self):
        self.model.load_state_dict(self._best_weights)
        if self.verbose:
            self.trace_func(
                "the current checkpoint is loaded into the model."
            )

    def reset_checkpoint(self):
        self._best_weights = None
        self._best_score: float = float("-inf" if self.maximize else "inf")
        if self.verbose:
            self.trace_func(
                "the current best checkpoint is reset with None."
            )

    def on_initialization(self, args: HandlerArguments):
        if self.model is None:
            self.model = args.model
        if self.init_from_checkpoint and os.path.isfile(self.save_path):
            self.load_from_checkpoint()

    def on_training_epoch_end(self, epoch: CurrentIterationStatus):
        metric_value = epoch.get_current_scores(self.monitor)[self.monitor]

        if self.is_value_better(metric_value):
            self._best_score = metric_value
            self._best_weights = self.model.state_dict()

    def on_training_end(self, stat: TrainerStatus):
        if self._best_weights is not None:
            self.model.load_state_dict(self._best_weights)
            if self.halt_into_checkpoint:
                self.save_into_checkpoint()

    def on_evaluation_begin(self, stat: TrainerStatus):
        if self.eval_with_best_model and self._best_weights is not None:
            self.model.load_state_dict(self._best_weights)

    def on_termination(self, stat: TrainerStatus):
        if self.halt_into_checkpoint and self._best_weights is not None:
            self.save_into_checkpoint()

    def on_stop_training_error(self, stat: TrainerStatus):
        if self.halt_into_checkpoint and self._best_weights is not None:
            self.save_into_checkpoint()
