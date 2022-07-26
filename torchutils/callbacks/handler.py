# Copyright © 2021 Chris Hughes
from typing import List, Optional
from .base import CallbackMethodNotImplementedError, TrainerCallback
from ..logging import LoggerInterface
from torchutils.trainer.utils import (
    TrainingArguments,
    EvaluatingArguments,
    Hyperparameter,
    IterationStatus,
    IterationInterface
)


def _foreach_callback_(method):
    def wrapped_method(self, *args, **kwargs):
        for callback in self.callbacks:
            self._callback = callback
            try:
                method(self, *args, **kwargs)
            except CallbackMethodNotImplementedError:
                continue
        self._callback = None
    return wrapped_method


def _foreach_callback_require_stat_(method):
    def wrapped_method(self, stat: IterationStatus, **kwargs):
        for callback in self.callbacks:
            self._callback = callback
            try:
                method(self, stat, **kwargs)
            except CallbackMethodNotImplementedError:
                continue
        self._callback = None
    return wrapped_method


def _foreach_callback_require_hparams_(method):
    def wrapped_method(self, hparams: Hyperparameter, **kwargs):
        for callback in self.callbacks:
            self._callback = callback
            try:
                method(self, hparams, **kwargs)
            except CallbackMethodNotImplementedError:
                continue
        self._callback = None
    return wrapped_method


class CallbackHandler:
    """
    The :class:`CallbackHandler` is responsible for calling a list of callbacks.
    This class calls the callbacks in the order that they are given.
    """
    slots = ['callbacks']

    def __init__(self, callbacks=None):
        self.callbacks: List[TrainerCallback] = []
        self._callback: Optional[TrainerCallback] = None
        if callbacks is not None:
            self.add_callbacks(callbacks)

    def remove_callbacks(self, callbacks):
        """
        Add a list of callbacks to the callback handler
        :param callbacks: a list of :class:`TrainerCallback`
        """
        for cb in callbacks:
            self.remove_callback(cb)

    def remove_callback(self, callback):
        """
        Add a callbacks to the callback handler
        :param callback: an instance of a subclass of :class:`TrainerCallback`
        """
        cb_class = callback if isinstance(
            callback, type) else callback.__class__
        if cb_class not in {c.__class__ for c in self.callbacks}:
            raise ValueError(
                f"You attempted to remove absensces of the callback {cb_class} to a single Trainer"
                f" The list of callbacks already present is\n: {self.callback_list}"
            )
        self.callbacks.remove(callback)

    def add_callbacks(self, callbacks):
        """
        Add a list of callbacks to the callback handler
        :param callbacks: a list of :class:`TrainerCallback`
        """
        for cb in callbacks:
            self.add_callback(cb)

    def add_callback(self, callback):
        """
        Add a callbacks to the callback handler
        :param callback: an instance of a subclass of :class:`TrainerCallback`
        """
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(
            callback, type) else callback.__class__
        if cb_class in {c.__class__ for c in self.callbacks}:
            raise ValueError(
                f"You attempted to add multiple instances of the callback {cb_class} to a single Trainer"
                f" The list of callbacks already present is\n: {self.callback_list}"
            )
        self.callbacks.append(cb)

    def __iter__(self):
        return self.callbacks

    def clear_callbacks(self):
        self.callbacks = []

    def __repr__(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    @_foreach_callback_
    def on_initialization(self, loggers: LoggerInterface):
        self._callback.on_initialization(loggers)

    @_foreach_callback_require_hparams_
    def on_training_begin(self, hparams: TrainingArguments):
        self._callback.on_training_begin(hparams)

    @_foreach_callback_require_stat_
    def on_training_epoch_begin(self, stat: IterationStatus):
        self._callback.on_training_epoch_begin(stat)

    @_foreach_callback_require_stat_
    def on_training_step_begin(self, stat: IterationStatus):
        self._callback.on_training_step_begin(stat)

    @_foreach_callback_require_stat_
    def on_training_step_end(self, batch: IterationInterface):
        self._callback.on_training_step_end(batch=batch)

    @_foreach_callback_
    def on_training_epoch_end(self, epoch: IterationInterface):
        self._callback.on_training_epoch_end(epoch=epoch)

    @_foreach_callback_require_stat_
    def on_training_end(self, stat: IterationStatus):
        self._callback.on_training_end(stat)

    @_foreach_callback_require_stat_
    def on_validation_run_begin(self, stat: IterationStatus):
        self._callback.on_validation_run_begin(stat)

    @_foreach_callback_require_stat_
    def on_validation_step_begin(self, stat: IterationStatus):
        self._callback.on_validation_step_begin(stat)

    @_foreach_callback_
    def on_validation_step_end(self, batch: IterationInterface):
        self._callback.on_validation_step_end(batch=batch)

    @_foreach_callback_
    def on_validation_run_end(self, epoch: IterationInterface):
        self._callback.on_validation_run_end(epoch=epoch)

    @_foreach_callback_require_hparams_
    def on_evaluation_run_begin(self, hparams: EvaluatingArguments):
        self._callback.on_evaluation_run_begin(hparams)

    @_foreach_callback_require_stat_
    def on_evaluation_step_begin(self, stat: IterationStatus):
        self._callback.on_evaluation_step_begin(stat)

    @_foreach_callback_
    def on_evaluation_step_end(self, batch: IterationInterface):
        self._callback.on_evaluation_step_end(batch=batch)

    @_foreach_callback_
    def on_evaluation_run_end(self, stat: IterationStatus):
        self._callback.on_evaluation_run_end(stat)

    @_foreach_callback_require_stat_
    def on_stop_training_error(self, stat: IterationStatus):
        self._callback.on_stop_training_error(stat)

    @_foreach_callback_require_stat_
    def on_termination(self, stat: IterationStatus):
        self._callback.on_termination(stat)
