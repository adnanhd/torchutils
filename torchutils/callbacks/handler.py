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

from .base import CallbackMethodNotImplementedError


class CallbackHandler:
    """
    The :class:`CallbackHandler` is responsible for calling a list of callbacks.
    This class calls the callbacks in the order that they are given.
    """
    slots = ['callbacks']

    def __init__(self, callbacks = None):
        self.callbacks = []
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

    def call_event(self, trainer, event, *args, **kwargs):
        """
        For each callback which has been registered, sequentially call the method corresponding to the
        given event.
        :param event: The event corresponding to the method to call on each callback
        :param args: a list of arguments to be passed to each callback
        :param kwargs: a list of keyword arguments to be passed to each callback
        """
        for callback in self.callbacks:
            try:
                callback.__getattribute__(event)(trainer=trainer, *args, **kwargs)
            except CallbackMethodNotImplementedError as e:
                continue



