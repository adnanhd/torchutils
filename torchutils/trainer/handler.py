# Import Handlees
from torchutils.metrics import TrainerMetric
from torchutils.logging import TrainerLogger
from torchutils.callbacks import TrainerCallback

# Import Handlers
from torchutils.metrics import MetricHandler
from torchutils.logging import LoggingHandler
from torchutils.callbacks import CallbackHandler

from .utils import LossTracker

# Import Arguments
from torchutils.utils.pydantic import (
    HandlerArguments, 
    TrainerStatus,
    StepResults,
    EpochResults,
    TrainerModel,
    TrainingArguments
)


#class HandlerArguments:
class TrainerHandler():
    # train_dl: TrainerDataLoaderArguments
    # valid_dl: TrainerDataLoaderArguments
    # test_dl: TrainerDataLoaderArguments
    # args: TrainingArguments
    # model: TrainerModel
    __slots__ = ['_loggers', '_metrics', '_callbacks',
            '_arguments', '_status', '_tracker']

    def __init__(self, 
            args: TrainingArguments,
            model: TrainerModel,
            status: TrainerStatus,
            **kwargs
        ):
        for key in ('train_dl', 'valid_dl', 'test_dl'):
            if kwargs.setdefault(key, None) is None: 
                kwargs.pop(key)
        self._loggers = LoggingHandler()
        self._metrics = MetricHandler()
        self._callbacks = CallbackHandler()
        self._arguments = HandlerArguments(args=args, model=model, status=status)
        self._status = status
        self._tracker = LossTracker()

    # Tracker manipulation functions
    def update(self, batch_loss, num_batchs=1):
        self._tracker.update(loss_batch_value=batch_loss, batch_size=num_batchs)

    def reset(self):
        self._tracker.reset()

    def compile(
            self,
            loggers = list(),
            metrics = list(),
            callbacks = list(),
            ):
        self._loggers.add_loggers(loggers)
        self._metrics.add_scores(metrics)
        self._callbacks.add_callbacks(callbacks)

    def decompile(
            self,
            loggers = list(), 
            metrics = list(), 
            callbacks = list(),
            ):
        self._loggers.remove_logger(loggers)
        self._metrics.remove_scores(metrics)
        self._callbacks.remove_callbacks(callbacks)

    def clear(self):
        self._loggers.clear_loggers()
        self._callbacks.clear_callbacks()
        self._metrics.clear_scores()

    def on_initialization(self):
        self._callbacks.on_initialization(self._arguments)

    def on_training_begin(self):
        self._callbacks.on_training_begin(self._status)

    def on_training_epoch_begin(self):
        self._callbacks.on_training_epoch_begin(self._status)

    def on_training_step_begin(self):
        #trainer.metrics.init(
        self._callbacks.on_training_step_begin(self._status)

    def on_training_step_end(self, x, y, y_pred):
        batch = StepResults(x=x, y_true=y, y_pred=y_pred)
        self._callbacks.on_training_step_end(batch)

    def on_training_epoch_end(self):
        epoch = EpochResults()
        self._callbacks.on_training_epoch_end(epoch)

    def on_training_end(self):
        self._callbacks.on_training_end(self._status)

    def on_validation_run_begin(self):
        self._callbacks.on_validation_run_begin(self._status)

    def on_validation_step_begin(self):
        self._callbacks.on_validation_step_begin(self._status)

    def on_validation_step_end(self, x, y, y_pred):
        batch = StepResults(x=x, y=y, y_pred=y_pred)
        self._callbacks.on_validation_step_end(batch)

    def on_validation_run_end(self):
        epoch = EpochResults()
        self._callbacks.on_validation_run_end(epoch)

    def on_evaluation_run_begin(self):
        self._callbacks.on_evaluation_run_begin(self._status)

    def on_evaluation_step_begin(self):
        self._callbacks.on_evaluation_step_begin(self._status)

    def on_evaluation_step_end(self, x, y, y_pred):
        batch = StepResults(x=x, y_true=y, y_pred=y_pred)
        self._callbacks.on_evaluation_step_end(batch)

    def on_evaluation_run_end(self):
        epoch = EpochResults()
        self._callbacks.on_evaluation_run_end(epoch)

    def on_stop_training_error(self):
        self._callbacks.on_stop_training_error(self._status)

    def on_termination(self):
        self._callbacks.on_termination(self._status)

