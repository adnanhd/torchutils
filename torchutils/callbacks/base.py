# Copyright © 2021 Chris Hughes
import abc
import logging
from ..logging import LoggerInterface
from torchutils.trainer.utils import (
    # IterationArguments,
    TrainingArguments,
    EvaluatingArguments,
    IterationStatus,
    IterationInterface
)


class StopTrainingError(Exception):
    """
    An exception which can be raised in order to stop a training run early.
    """
    pass


class CallbackMethodNotImplementedError(Exception):
    pass


class TrainerCallback(abc.ABC):
    """
    The abstract base class to be subclassed when creating new callbacks.
    """
    __slots__ = ['__logger__']

    def __init__(self):
        self.__logger__: logging.Logger = logging.getLogger(
            self.__class__.__name__
        )

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '__logger__'):
            return logging.getLogger(self.__class__.__name__)
        else:
            return self.__logger__

    def on_initialization(self, loggers: LoggerInterface):
        """
        Event called at the end of trainer initialisation.
        """
        raise CallbackMethodNotImplementedError

    def on_training_begin(self, hparams: TrainingArguments):
        """
        Event called at the start of training run.
        :param num_epochs total number of epochs at the most 
        :param batch_size the number of samples in a training batch 
        :param step_size the number of batches
        """
        raise CallbackMethodNotImplementedError

    def on_training_epoch_begin(self, stat: IterationStatus):
        """
        Event called at the beginning of a training epoch.
        :param epoch
        """
        raise CallbackMethodNotImplementedError

    def on_training_step_begin(self, stat: IterationStatus):
        """
        Event called at the beginning of a training step.
        """
        raise CallbackMethodNotImplementedError

    def on_training_step_end(self, batch: IterationInterface):
        """
        Event called at the end of a training step.
        :param batch: the current batch of training data
        :param batch_output: the outputs returned by :meth:`pytorch_accelerated.trainer.Trainer.calculate_train_batch_loss`
        """
        raise CallbackMethodNotImplementedError

    def on_training_epoch_end(self, epoch: IterationInterface):
        """
        Event called at the end of a training epoch.
        """
        raise CallbackMethodNotImplementedError

    def on_training_end(self, stat: IterationStatus):
        """
        Event called at the end of training run.
        """
        raise CallbackMethodNotImplementedError

    def on_validation_run_begin(self, stat: IterationStatus):
        """
        Event called at the beginning of an evaluation epoch.
        """
        raise CallbackMethodNotImplementedError

    def on_validation_step_begin(self, stat: IterationStatus):
        """
        Event called at the beginning of a evaluation step.
        """
        raise CallbackMethodNotImplementedError

    def on_validation_step_end(self, batch: IterationInterface):
        """
        Event called at the end of an evaluation step.
        :param batch: the current batch of evaluation data
        :param batch_output: the outputs returned by :meth:`pytorch_accelerated.trainer.Trainer.calculate_eval_batch_loss`
        """
        raise CallbackMethodNotImplementedError

    def on_validation_run_end(self, epoch: IterationInterface):
        """
        Event called at the start of an evaluation run.
        """
        raise CallbackMethodNotImplementedError

    def on_training_valid_end(self, stat: IterationStatus):
        """
        Event called during a training run after both training and evaluation epochs have been completed.
        """
        raise CallbackMethodNotImplementedError

    def on_evaluation_run_begin(self, hparams: EvaluatingArguments):
        """
        Event called at the start of an evaluation run.
        """
        raise CallbackMethodNotImplementedError

    def on_evaluation_step_begin(self, stat: IterationStatus):
        """
        Event called at the beginning of a evaluation step.
        """
        raise CallbackMethodNotImplementedError

    def on_evaluation_step_end(self, batch: IterationInterface):
        """
        Event called at the end of an evaluation step.
        :param batch: the current batch of evaluation data
        :param batch_output: the outputs returned by :meth:`pytorch_accelerated.trainer.Trainer.calculate_eval_batch_loss`
        """
        raise CallbackMethodNotImplementedError

    def on_evaluation_run_end(self, epoch: IterationInterface):
        """
        Event called at the end of an evaluation run.
        """
        raise CallbackMethodNotImplementedError

    def on_stop_training_error(self, stat: IterationStatus):
        """
        Event called when a stop training error is raised
        """
        raise CallbackMethodNotImplementedError

    def on_termination(self, stat: IterationStatus):
        raise CallbackMethodNotImplementedError

    # def __getattr__(self, item):
    #    try:
    #        return super().__getattr__(item)
    #    except AttributeError:
    #        raise CallbackMethodNotImplementedError
