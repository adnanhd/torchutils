# Copyright © 2021 Chris Hughes
from abc import ABC
from typing import Optional
from torchutils.logging import LoggerProxy, LoggerHandler
from torchutils.utils.pydantic import (
    HandlerArguments,
    TrainerStatus,
    CurrentIterationStatus
)


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
    __slots__ = ['__log__']

    def __init__(self):
        self.__log__: Optional[LoggerProxy] = None

    @property
    def _log(self) -> LoggerProxy:
        if self.__log__ is None:
            self.__log__ = LoggerHandler.getProxy()
        return self.__log__

    def on_initialization(self, args: HandlerArguments):
        """
        Event called at the end of trainer initialisation.
        """
        raise CallbackMethodNotImplementedError

    def on_training_begin(self, stat: TrainerStatus):
        """
        Event called at the start of training run.
        :param num_epochs total number of epochs at the most 
        :param batch_size the number of samples in a training batch 
        :param step_size the number of batches
        """
        raise CallbackMethodNotImplementedError

    def on_training_epoch_begin(self, stat: TrainerStatus):
        """
        Event called at the beginning of a training epoch.
        :param epoch
        """
        raise CallbackMethodNotImplementedError

    def on_training_step_begin(self, stat: TrainerStatus):
        """
        Event called at the beginning of a training step.
        """
        raise CallbackMethodNotImplementedError

    def on_training_step_end(self, batch: CurrentIterationStatus):
        """
        Event called at the end of a training step.
        :param batch: the current batch of training data
        :param batch_output: the outputs returned by :meth:`pytorch_accelerated.trainer.Trainer.calculate_train_batch_loss`
        """
        raise CallbackMethodNotImplementedError

    def on_training_epoch_end(self, epoch: CurrentIterationStatus):
        """
        Event called at the end of a training epoch.
        """
        raise CallbackMethodNotImplementedError

    def on_training_end(self, stat: TrainerStatus):
        """
        Event called at the end of training run.
        """
        raise CallbackMethodNotImplementedError

    def on_validation_run_begin(self, stat: TrainerStatus):
        """
        Event called at the beginning of an evaluation epoch.
        """
        raise CallbackMethodNotImplementedError

    def on_validation_step_begin(self, stat: TrainerStatus):
        """
        Event called at the beginning of a evaluation step.
        """
        raise CallbackMethodNotImplementedError

    def on_validation_step_end(self, batch: CurrentIterationStatus):
        """
        Event called at the end of an evaluation step.
        :param batch: the current batch of evaluation data
        :param batch_output: the outputs returned by :meth:`pytorch_accelerated.trainer.Trainer.calculate_eval_batch_loss`
        """
        raise CallbackMethodNotImplementedError

    def on_validation_run_end(self, epoch: CurrentIterationStatus):
        """
        Event called at the start of an evaluation run.
        """
        raise CallbackMethodNotImplementedError

    def on_training_valid_end(self, stat: TrainerStatus):
        """
        Event called during a training run after both training and evaluation epochs have been completed.
        """
        raise CallbackMethodNotImplementedError

    def on_evaluation_run_begin(self, stat: TrainerStatus):
        """
        Event called at the start of an evaluation run.
        """
        raise CallbackMethodNotImplementedError

    def on_evaluation_step_begin(self, stat: TrainerStatus):
        """
        Event called at the beginning of a evaluation step.
        """
        raise CallbackMethodNotImplementedError

    def on_evaluation_step_end(self, batch: CurrentIterationStatus):
        """
        Event called at the end of an evaluation step.
        :param batch: the current batch of evaluation data
        :param batch_output: the outputs returned by :meth:`pytorch_accelerated.trainer.Trainer.calculate_eval_batch_loss`
        """
        raise CallbackMethodNotImplementedError

    def on_evaluation_run_end(self, epoch: CurrentIterationStatus):
        """
        Event called at the end of an evaluation run.
        """
        raise CallbackMethodNotImplementedError

    def on_stop_training_error(self, stat: TrainerStatus):
        """
        Event called when a stop training error is raised
        """
        raise CallbackMethodNotImplementedError

    def on_termination(self, stat: TrainerStatus):
        raise CallbackMethodNotImplementedError

    # def __getattr__(self, item):
    #    try:
    #        return super().__getattr__(item)
    #    except AttributeError:
    #        raise CallbackMethodNotImplementedError
