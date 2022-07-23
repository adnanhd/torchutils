# Copyright © 2021 Chris Hughes
import os
import copy
import torch
import logging
import warnings
from .base import TrainerCallback
from typing import Callable, List
from ..models.utils import TrainerModel
from ..trainer.utils import (
    IterationArguments,
    IterationStatus,
    IterationInterface
)


def profiler(fn):
    def wrapped_fn(self, *args, **kwargs):
        self.logger.debug(f'{fn.__name__} is called')
        return fn(self, *args, **kwargs)
    return wrapped_fn


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
                 monitor="Loss",
                 trainer_model: TrainerModel = None,
                 save_path='model_checkpoint.ckpt',
                 maximize_score: bool = False,
                 delta: float = 0.0,
                 # TODO: replace this param with TrainerLogger
                 handlers: List[logging.Handler] = [],
                 verbose: bool = False,
                 load_back_per_epochs: int = 0,
                 init_from_checkpoint: bool = False,
                 halt_into_checkpoint: bool = False,
                 save_only_best_model: bool = True,
                 eval_with_best_model: bool = False,
                 ):
        super().__init__()
        self.monitor: str = monitor
        self.model: TrainerModel = trainer_model
        self.delta: float = delta
        self.maximize: bool = maximize_score
        self.save_path: str = save_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.verbose: bool = verbose
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARNING)
        for handler in handlers:
            self.logger.addHandler(handler)

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
            def on_training_epoch_begin(self, status: IterationStatus):
                if self._best_weights is not None and \
                        status.current_epoch % self.load_back_per_epochs == 0:
                    self._put_checkpoint_into_model()
            self.on_training_epoch_begin = on_training_epoch_begin

    def is_value_better(self, metric_value: float) -> bool:
        # if aim is to maximize score, then check whether metric_value is
        # greater then our best score with a margin/confidency of delta
        # otherwise, then check whether metric_value is lower then our best
        # result with a margin/confidency of delta
        return self.maximize \
            and metric_value > (1 + self.delta) * self._best_score \
            or metric_value < (1 - self.delta) * self._best_score

    @profiler
    def _save_into_filesystem(self) -> None:
        if self.save_only_best_model and os.path.isfile(self.save_path):
            checkpoint = torch.load(self.save_path)
            try:
                best_score = checkpoint['scores'][self.monitor]
                self.logger.debug(f'best score is set to {best_score}')
            except KeyError:
                save_model = True
                self.logger.debug(
                    'best score could not found in the checkpoint')
            else:
                save_model = not self.is_value_better(metric_value=best_score)
                self.logger.debug(
                    f'best score is not better than {self._best_score}')
        else:
            save_model = True
            self.logger.debug(
                f'save_only_best_model is {self.save_only_best_model}')
            self.logger.debug(
                f'{self.save_path} existence is {os.path.isfile(self.save_path)}')
        if save_model:
            checkpoint = {'state_dict': self._best_weights,
                          'scores': {self.monitor: self._best_score}}
            self.logger.info(
                f'the model with state_dicts {checkpoint["state_dict"].keys()} '
                f'and with scores {checkpoint["scores"]} is saved into {self.save_path}.'
            )
            torch.save(checkpoint, self.save_path)

    @profiler
    def _load_from_filesystem(self):
        if os.path.isfile(self.save_path):
            checkpoint = torch.load(f=self.save_path,
                                    map_location=self.model.device)
            try:
                best_score = checkpoint['scores'][self.monitor]
            except KeyError:
                load_model = False
                best_score = None
            else:
                load_model = self.is_value_better(best_score)

            if not self.save_only_best_model:
                load_model = True

            self.logger.debug(f'load_model: {load_model}')
            if load_model:
                self._reset_checkpoints()
                self._best_weights = checkpoint['state_dict']
                if best_score is not None:
                    self._best_score = best_score
                self.logger.info(
                    f'the model with the models {checkpoint["state_dict"].keys()} '
                    f'and with scores {checkpoint["scores"]} is loaded from {self.save_path}.'
                )
        else:
            warnings.warn(
                "ModelCheckpoint._load_from_filesystem: file could not "
                f"found in {self.save_path}")

    @profiler
    def _get_checkpoint_from_model(self, model_score):
        self._best_score = model_score
        self._best_weights = self.model.state_dict()
        self.logger.info(
            f"A newer checkpoint with score {model_score} "
            f"is obtained from the model {self._best_weights.keys()}."
        )

    @profiler
    def _put_checkpoint_into_model(self):
        self.model.load_state_dict(copy.deepcopy(self._best_weights))
        self.logger.info(
            f"the current checkpoint with score {self._best_score} "
            f"is loaded into the model {self._best_weights.keys()}."
        )

    @profiler
    def _reset_checkpoints(self):
        self._best_weights = None
        self._best_score: float = float("-inf" if self.maximize else "inf")
        self.logger.info(
            "the current best checkpoint is reset with None."
        )

    @profiler
    def on_initialization(self, args: IterationArguments):
        if self.model is None:
            self.model = args.model
            self.logger.debug('self.model is set')
        self.logger.debug(
            f'init from checkpoint is {self.init_from_checkpoint}')
        self.logger.debug(
            f'{self.save_path} existence is {os.path.isfile(self.save_path)}')
        if self.init_from_checkpoint and os.path.isfile(self.save_path):
            self._load_from_filesystem()

    def on_training_epoch_end(self, epoch: IterationInterface):
        metric_value = epoch.get_current_scores(self.monitor)[self.monitor]

        self.logger.debug(f'Epoch {epoch.status.current_epoch} of '
                          f'{epoch.hparams.num_epochs}: '
                          f'current_value: {self._best_score} # '
                          f'compared value: {metric_value}')

        if self.is_value_better(metric_value):
            self._get_checkpoint_from_model(metric_value)

    def on_training_end(self, stat: IterationStatus):
        if self._best_weights is not None:
            if self.halt_into_checkpoint:
                self._save_into_filesystem()
            self._put_checkpoint_into_model()

    def on_evaluation_begin(self, stat: IterationStatus):
        self.logger.debug(
            f'best weights absence is {self._best_weights is None}')
        if self.eval_with_best_model and self._best_weights is not None:
            self._put_checkpoint_into_model()

    @profiler
    def on_stop_training_error(self, stat: IterationStatus):
        if self._best_weights is not None:
            if self.halt_into_checkpoint:
                self._save_into_filesystem()
            self._put_checkpoint_into_model()
