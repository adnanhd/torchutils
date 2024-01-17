from .callback import TrainerCallback, StopTrainingException
from .callback import CallbackMethodNotImplemented
import typing
import logging


def evettriggermethod(method):
    def evettriggerdecorator(self, *args, **kwargs):
        stop_training = False
        for callback in self.callbacks:
            try:
                getattr(callback, method.__name__)(*args, **kwargs)
            except CallbackMethodNotImplemented:
                continue
            except StopTrainingException:
                stop_training = True

        if stop_training:
            raise StopTrainingException

    return evettriggerdecorator


class CallbackHandler:
    """
    The :class:`CallbackHandler` is responsible for calling a list of callbacks.
    This class calls the callbacks in the order that they are given.
    """
    __slots__ = ['callbacks']

    def __init__(self, callbacks=list()):
        self.callbacks: typing.List[TrainerCallback] = callbacks

    def add_handlers(self, handlers: typing.List[logging.Handler] = list()):
        for cback in self.callbacks:
            cback.add_handlers(handlers)

    def remove_handlers(self, handlers: typing.List[logging.Handler] = list()):
        for cback in self.callbacks:
            cback.remove_handlers(handlers)

    def attach_score_dict(self, score_dict: typing.Dict[str, float]):
        for cback in self.callbacks:
            cback.attach_score_dict(score_dict)

    def detach_score_dict(self):
        for cback in self.callbacks:
            cback.detach_score_dict()

    def __iter__(self):
        return self.callbacks

    def __repr__(self):
        callbacks = ", ".join(cb.__class__.__name__ for cb in self.callbacks)
        return self.__class__.__name__ + '(' + callbacks + ')'

    @evettriggermethod
    def on_initialization(self): pass

    @evettriggermethod
    def on_training_begin(self, params): pass

    @evettriggermethod
    def on_training_epoch_begin(self, epoch_index: int): pass

    @evettriggermethod
    def on_training_step_begin(self, batch_index: int): pass

    @evettriggermethod
    def on_training_step_end(self, batch_index, batch, batch_output): pass

    @evettriggermethod
    def on_training_epoch_end(self): pass

    @evettriggermethod
    def on_training_end(self): pass

    @evettriggermethod
    def on_validation_run_begin(self, epoch_index: int): pass

    @evettriggermethod
    def on_validation_step_begin(self, batch_index: int): pass

    @evettriggermethod
    def on_validation_step_end(self, batch_index, batch, batch_output): pass

    @evettriggermethod
    def on_validation_run_end(self): pass

    @evettriggermethod
    def on_evaluation_run_begin(self, params): pass

    @evettriggermethod
    def on_evaluation_step_begin(self): pass

    @evettriggermethod
    def on_evaluation_step_end(self, batch_index, batch, batch_output): pass

    @evettriggermethod
    def on_evaluation_run_end(self): pass

    @evettriggermethod
    def on_stop_training_error(self): pass

    @evettriggermethod
    def on_termination(self): pass
