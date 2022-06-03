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
from typing import List, Optional

from abc import abstractmethod
from .base import CallbackMethodNotImplementedError, TrainerCallback
from torchutils.utils.pydantic import (
        HandlerArguments,
        TrainerStatus,
        EpochResults,
        StepResults
)

class CallbackHandler:
    """
    The :class:`CallbackHandler` is responsible for calling a list of callbacks.
    This class calls the callbacks in the order that they are given.
    """
    slots = ['callbacks']

    def __init__(self, callbacks = None):
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
        cb_class = callback if isinstance(callback, type) else callback.__class__
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
        cb_class = callback if isinstance(callback, type) else callback.__class__
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

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def __repr__(self):
        return self.callback_list
   
    def run_begin(self, args: HandlerArguments):
        pass

    def epoch_begin(self, stat: TrainerStatus):
        pass

    def step_begin(self, stat: TrainerStatus):
        pass

    def step_end(self, batch: StepResults):
        pass

    def epoch_end(self, epoch: EpochResults):
        pass

    def run_end(self, stat: TrainerStatus):
        pass


def foreach_callback(cls, method):
    def wrapped_method(self, *args, **kwargs):
        for callback in self.callbacks:
            self._callback = callback
            try:
                method(*args, **kwargs)
            except CallbackMethodNotImplementedError:
                continue
        self._callback = None
    return wrapped_method


class Train_CallbackHandler(CallbackHandler):
    @foreach_callback
    def run_begin(self, args: HandlerArguments):
        self._callback.on_training_begin(args)

    @foreach_callback
    def epoch_begin(self, stat: TrainerStatus):
        self._callback.on_training_epoch_begin(stat)

    @foreach_callback
    def step_begin(self, stat: TrainerStatus):
        self._callback.on_training_step_begin(stat)

    @foreach_callback
    def step_end(self, batch: StepResults):
        self._callback.on_training_step_end(batch)

    @foreach_callback
    def epoch_end(self, epoch: EpochResults):
        self._callback.on_training_epoch_end(epoch)

    @foreach_callback
    def run_end(self, stat: TrainerStatus):
        self._callback.on_training_end(stat)


class Valid_CallbackHandler(CallbackHandler):
    @foreach_callback
    def epoch_begin(self, stat: TrainerStatus):
        self._callback.on_validation_run_begin(stat)

    @foreach_callback
    def step_begin(self, stat: TrainerStatus):
        self._callback.on_validation_step_begin(stat)

    @foreach_callback
    def step_end(self, batch: StepResults):
        self._callback.on_validation_step_end(batch)

    @foreach_callback
    def epoch_end(self, epoch: EpochResults):
        self._callback.on_validation_run_end(epoch)


class Eval_CallbackHandler(CallbackHandler):
    @foreach_callback
    def run_begin(self, args: HandlerArguments):
        self._callback.on_evaluation_run_begin(stat)

    @foreach_callback
    def step_begin(self, stat: TrainerStatus):
        self._callback.on_evaluation_step_begin(stat)

    @foreach_callback
    def step_end(self, batch: StepResults):
        self._callback.on_evaluation_step_end(batch)

    @foreach_callback
    def run_end(self, epoch: EpochResults):
        self._callback.on_evaluation_step_end(epoch)

