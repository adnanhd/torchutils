from .callback import TrainerCallback, StopTraining, CallbackMethodNotImplementedError
import typing


def evettriggermethod(method):
    def evettriggerdecorator(self, *args, **kwargs):
        stop_training = False
        for callback in self.callbacks:
            try:
                getattr(callback, method.__name__)(*args, **kwargs)
            except CallbackMethodNotImplementedError:
                continue
            except StopTraining:
                stop_training = True
        
        if stop_training:
            raise StopTraining

    return evettriggerdecorator

class CallbackHandler:
    """
    The :class:`CallbackHandler` is responsible for calling a list of callbacks.
    This class calls the callbacks in the order that they are given.
    """
    __slots__ = ['callbacks']

    def __init__(self, callbacks=None):
        self.callbacks: typing.List[TrainerCallback] = []
        if callbacks is not None:
            self.callbacks.extend(callbacks)

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
    def on_training_epoch_begin(self): pass

    @evettriggermethod
    def on_training_step_begin(self): pass

    @evettriggermethod
    def on_training_step_end(self, batch_index, batch, batch_output): pass

    @evettriggermethod
    def on_training_epoch_end(self): pass

    @evettriggermethod
    def on_training_end(self): pass

    @evettriggermethod
    def on_validation_run_begin(self): pass

    @evettriggermethod
    def on_validation_step_begin(self): pass

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