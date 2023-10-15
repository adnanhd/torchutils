# Copyright © 2021 Chris Hughes
import abc
import logging
import typing


def profiler(fn):
    def wrapped_fn(self, *args, **kwargs):
        self.logger.debug(f'{fn.__name__} is called')
        return fn(self, *args, **kwargs)
    return wrapped_fn


class StopTrainingException(Exception):
    """
    An exception which can be raised in order to stop a training run early.
    """
    pass


class CallbackMethodNotImplemented(Exception):
    pass


class TrainerCallback(abc.ABC):
    """
    The abstract base class to be subclassed when creating new callbacks.
    """

    def __init__(self, level=logging.INFO, verbose=True):
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        self.logger.setLevel(level if verbose else logging.FATAL)
        self.scores = dict()

    @typing.final
    def add_handlers(self, handlers):
        for hdlr in handlers:
            self.logger.addHandler(hdlr)

    @typing.final
    def remove_handlers(self, handlers):
        for hdlr in handlers:
            self.logger.removeHandler(hdlr)

    @typing.final
    def attach_score_dict(self, score_dict):
        self.scores = score_dict

    @typing.final
    def detach_score_dict(self):
        self.scores = dict()

    def on_initialization(self):
        """
        Event called at the end of trainer initialisation.
        """
        raise CallbackMethodNotImplemented

    def on_training_begin(self, kwargs):
        """
        Event called at the start of training run.
        :param num_epochs total number of epochs at the most
        :param batch_size the number of samples in a training batch
        :param step_size the number of batches
        """
        raise CallbackMethodNotImplemented

    def on_training_epoch_begin(self):
        """
        Event called at the beginning of a training epoch.
        :param epoch
        """
        raise CallbackMethodNotImplemented

    def on_training_step_begin(self):
        """
        Event called at the beginning of a training step.
        """
        raise CallbackMethodNotImplemented

    def on_training_step_end(self, batch_index, batch, batch_output):
        """
        Event called at the end of a training step.
        :param batch: the current batch of training data
        :param batch_output: the outputs returned by :meth:
            `models.TrainerModel.forward_pass`
        """
        raise CallbackMethodNotImplemented

    def on_training_epoch_end(self):
        """
        Event called at the end of a training epoch.
        """
        raise CallbackMethodNotImplemented

    def on_training_end(self):
        """
        Event called at the end of training run.
        """
        raise CallbackMethodNotImplemented

    def on_validation_run_begin(self):
        """
        Event called at the beginning of an evaluation epoch.
        """
        raise CallbackMethodNotImplemented

    def on_validation_step_begin(self):
        """
        Event called at the beginning of a evaluation step.
        """
        raise CallbackMethodNotImplemented

    def on_validation_step_end(self, batch_index, batch, batch_output):
        """
        Event called at the end of an evaluation step.
        :param batch: the current batch of evaluation data
        :param batch_output: the outputs returned by :meth:
            `models.TrainerModel.forward_pass`
        """
        raise CallbackMethodNotImplemented

    def on_validation_run_end(self):
        """
        Event called at the start of an evaluation run.
        """
        raise CallbackMethodNotImplemented

    def on_training_valid_end(self):
        """
        Event called during a training run after
        both training and evaluation epochs have been completed.
        """
        raise CallbackMethodNotImplemented

    def on_evaluation_run_begin(self, kwargs):
        """
        Event called at the start of an evaluation run.
        """
        raise CallbackMethodNotImplemented

    def on_evaluation_step_begin(self):
        """
        Event called at the beginning of a evaluation step.
        """
        raise CallbackMethodNotImplemented

    def on_evaluation_step_end(self, batch_index, batch, batch_output):
        """
        Event called at the end of an evaluation step.
        :param batch: the current batch of evaluation data
        :param batch_output: the outputs returned by :meth:
            `models.TrainerModel.forward_pass`
        """
        raise CallbackMethodNotImplemented

    def on_evaluation_run_end(self):
        """
        Event called at the end of an evaluation run.
        """
        raise CallbackMethodNotImplemented

    def on_stop_training_error(self):
        """
        Event called when a stop training error is raised
        """
        raise CallbackMethodNotImplemented

    def on_termination(self):
        raise CallbackMethodNotImplemented
