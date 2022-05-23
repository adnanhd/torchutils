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
import dataclasses



class StopTrainingError(Exception):
    """
    An exception which can be raised in order to stop a training run early.
    """

    pass


class CallbackMethodNotImplementedError(Exception):
    pass


class TrainerCallback(ABC):
    """
    The abstract base class to be subclassed when creating new callbacks.
    """

    def on_initialization(self, trainer, **kwargs):
        """
        Event called at the end of trainer initialisation.
        """
        pass

    def on_training_begin(self, trainer, **kwargs):
        """
        Event called at the start of training run.
        :param num_epochs total number of epochs at the most 
        :param batch_size the number of samples in a training batch 
        :param step_size the number of batches
        """
        pass

    def on_training_epoch_begin(self, trainer, **kwargs):
        """
        Event called at the beginning of a training epoch.
        :param epoch
        """
        pass

    def on_training_step_begin(self, trainer, **kwargs):
        """
        Event called at the beginning of a training step.
        """
        pass

    def on_training_step_end(self, trainer, batch, batch_output, **kwargs):
        """
        Event called at the end of a training step.
        :param batch: the current batch of training data
        :param batch_output: the outputs returned by :meth:`pytorch_accelerated.trainer.Trainer.calculate_train_batch_loss`
        """
        pass

    def on_training_epoch_end(self, trainer, **kwargs):
        """
        Event called at the end of a training epoch.
        """
        pass

    def on_training_end(self, trainer, **kwargs):
        """
        Event called at the end of training run.
        """
        pass

    def on_validation_run_begin(self, trainer, **kwargs):
        """
        Event called at the beginning of an evaluation epoch.
        """
        pass

    def on_validation_step_begin(self, trainer, **kwargs):
        """
        Event called at the beginning of a evaluation step.
        """
        pass

    def on_validation_step_end(self, trainer, batch, batch_output, **kwargs):
        """
        Event called at the end of an evaluation step.
        :param batch: the current batch of evaluation data
        :param batch_output: the outputs returned by :meth:`pytorch_accelerated.trainer.Trainer.calculate_eval_batch_loss`
        """
        pass
    
    def on_validation_run_end(self, trainer, **kwargs):
        """
        Event called at the start of an evaluation run.
        """
        pass

    def on_training_valid_end(self, trainer, **kwargs):
        """
        Event called during a training run after both training and evaluation epochs have been completed.
        """
        pass

    def on_evaluation_run_begin(self, trainer, **kwargs):
        """
        Event called at the start of an evaluation run.
        """
        pass

    def on_evaluation_step_begin(self, trainer, **kwargs):
        """
        Event called at the beginning of a evaluation step.
        """
        pass

    def on_evaluation_step_end(self, trainer, batch, batch_output, **kwargs):
        """
        Event called at the end of an evaluation step.
        :param batch: the current batch of evaluation data
        :param batch_output: the outputs returned by :meth:`pytorch_accelerated.trainer.Trainer.calculate_eval_batch_loss`
        """
        pass

    def on_evaluation_run_end(self, trainer, **kwargs):
        """
        Event called at the end of an evaluation run.
        """
        pass

    def on_stop_training_error(self, trainer, **kwargs):
        """
        Event called when a stop training error is raised
        """
        pass

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            raise CallbackMethodNotImplementedError

