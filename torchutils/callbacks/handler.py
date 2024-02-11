from .callback import TrainerCallback, StopTrainingException
from .callback import CallbackMethodNotImplemented
import typing
import logging
from .._dev_utils import MeterModelContainer


def evettriggermethod(method):
    def evettriggerdecorator(self, *args, **kwargs):
        stop_training = False
        for callback in self.elements:
            try:
                getattr(callback, method.__name__)(*args, **kwargs)
            except CallbackMethodNotImplemented:
                continue
            except StopTrainingException:
                stop_training = True
            except TypeError as e:
                lgr = logging.getLogger(CallbackHandler.__module__ + '.' + CallbackHandler.__class__.__name__)
                lgr.fatal(f'TypeError {callback.__class__.__name__}.{method.__name__}(): {e}')

        if stop_training:
            raise StopTrainingException

    return evettriggerdecorator


class CallbackHandler(MeterModelContainer):
    """
    The :class:`CallbackHandler` is responsible for calling a list of callbacks.
    This class calls the callbacks in the order that they are given.
    """
    def __init__(self, callbacks=list()):
        super().__init__(elements=callbacks)

    def __repr__(self):
        callbacks = ", ".join(cb.__class__.__name__ for cb in self.elements)
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
    def on_training_step_end(self, batch_index: int): pass

    @evettriggermethod
    def on_training_epoch_end(self): pass

    @evettriggermethod
    def on_training_end(self): pass

    @evettriggermethod
    def on_validation_run_begin(self, epoch_index: int): pass

    @evettriggermethod
    def on_validation_step_begin(self, batch_index: int): pass

    @evettriggermethod
    def on_validation_step_end(self, batch_index: int): pass

    @evettriggermethod
    def on_validation_run_end(self): pass

    @evettriggermethod
    def on_evaluation_run_begin(self, params): pass

    @evettriggermethod
    def on_evaluation_step_begin(self, batch_index: int): pass

    @evettriggermethod
    def on_evaluation_step_end(self, batch_index: int): pass

    @evettriggermethod
    def on_evaluation_run_end(self): pass

    @evettriggermethod
    def on_stop_training_error(self): pass

    @evettriggermethod
    def on_termination(self): pass
