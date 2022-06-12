# Import Handlees
from torchutils.metrics import TrainerMetric
from torchutils.logging import TrainerLogger
from torchutils.callbacks import TrainerCallback

# Import Handlers
from torchutils.metrics import MetricHandler
from torchutils.logging import LoggingHandler
from torchutils.callbacks import CallbackHandler

from .utils import ScoreTrackerHandler
from typing import Union, Optional, List

# Import Arguments
from torchutils.utils.pydantic import (
    HandlerArguments,
    TrainerStatus,
    StepResults,
    EpochResults,
    TrainerModel,
    TrainingArguments,
    EvaluatingArguments,
    TrainerDataLoader
)


# class HandlerArguments:
class TrainerHandler():
    # train_dl: TrainerDataLoaderArguments
    # valid_dl: TrainerDataLoaderArguments
    # eval_dl: TrainerDataLoaderArguments
    # args: TrainingArguments
    # model: TrainerModel
    __slots__ = ['arguments', '_tracker',
                 '_loggers', '_metrics',
                 '_callbacks', '__args_setter__']

    def __init__(self,
                 model: TrainerModel,
                 status_ptr: List[TrainerStatus],
                 **kwargs
                 ):
        for key in ('train_dl', 'valid_dl', 'eval_dl'):
            if kwargs.setdefault(key, None) is None:
                kwargs.pop(key)
        self._loggers = LoggingHandler()
        self._metrics = MetricHandler()
        self._callbacks = CallbackHandler()
        self.arguments = HandlerArguments(model=model, status_ptr=status_ptr)
        self.__args_setter__ = self.arguments.set_arguments()

    def set_arguments(self,
                      args: Union[TrainingArguments, EvaluatingArguments],
                      eval_dl: Optional[TrainerDataLoader] = None,
                      train_dl: Optional[TrainerDataLoader] = None,
                      valid_dl: Optional[TrainerDataLoader] = None,
                      ):
        dataloaders = {'train_dl': train_dl,
                       'valid_dl': valid_dl, 'eval_dl': eval_dl}
        dataloaders = {key: dataloader for key, dataloader
                       in dataloaders.items() if dataloader is not None}
        self.__args_setter__(args, dataloaders)

    @property
    def status(self):
        return self.arguments.status

    # Tracker manipulation functions
    def update(self, batch_loss, num_batchs=1):
        self.tracker.update(loss_batch_value=batch_loss, batch_size=num_batchs)

    def reset(self):
        self.tracker.reset()

    def compile(
            self,
            loggers=list(),
            metrics=list(),
            callbacks=list(),
    ):
        self._loggers.add_loggers(loggers)
        self._metrics.add_scores(metrics)
        self._callbacks.add_callbacks(callbacks)

    def decompile(
            self,
            loggers=list(),
            metrics=list(),
            callbacks=list(),
    ):
        self._loggers.remove_logger(loggers)
        self._metrics.remove_scores(metrics)
        self._callbacks.remove_callbacks(callbacks)

    def clear(self):
        self._loggers.clear_loggers()
        self._callbacks.clear_callbacks()
        self._metrics.clear_scores()

    def on_initialization(self):
        self._loggers.initialize(self.arguments)
        self._callbacks.on_initialization(self.arguments)

    def on_training_begin(self):
        self._loggers.model(self.arguments.model)
        self._callbacks.on_training_begin(self.status)

    def on_training_epoch_begin(self):
        self._callbacks.on_training_epoch_begin(self.status)

    def on_training_step_begin(self):
        self._callbacks.on_training_step_begin(self.status)

    def on_training_step_end(self, x, y, y_pred):
        batch = StepResults(x=x, y_true=y, y_pred=y_pred)
        self._metrics.set_scores_values(x=x, y=y, y_pred=y_pred)
        self._callbacks.on_training_step_end(batch)
        metrics = self._metrics.get_score_values()
        metrics['loss'] = self.tracker.average
        self._loggers.score(**metrics)

    def on_training_epoch_end(self):
        epoch = EpochResults()
        self._callbacks.on_training_epoch_end(epoch)

    def on_training_end(self):
        self._callbacks.on_training_end(self.status)

    def on_validation_run_begin(self):
        self._callbacks.on_validation_run_begin(self.status)

    def on_validation_step_begin(self):
        self._callbacks.on_validation_step_begin(self.status)

    def on_validation_step_end(self, x, y, y_pred):
        batch = StepResults(x=x, y_true=y, y_pred=y_pred)
        self._metrics.set_scores_values(x=x, y=y, y_pred=y_pred)
        self._callbacks.on_validation_step_end(batch)

    def on_validation_run_end(self):
        epoch = EpochResults()
        self._callbacks.on_validation_run_end(epoch)
        metrics = self._metrics.get_score_values()
        metrics['val_loss'] = self.tracker.average
        self._loggers.score(**metrics)

    def on_evaluation_run_begin(self):
        self._callbacks.on_evaluation_run_begin(self.status)

    def on_evaluation_step_begin(self):
        self._callbacks.on_evaluation_step_begin(self.status)

    def on_evaluation_step_end(self, x, y, y_pred):
        batch = StepResults(x=x, y_true=y, y_pred=y_pred)
        self._metrics.set_scores_values(x=x, y=y, y_pred=y_pred)
        self._callbacks.on_evaluation_step_end(batch)

    def on_evaluation_run_end(self):
        epoch = EpochResults()
        self._callbacks.on_evaluation_run_end(epoch)
        metrics = self._metrics.get_score_values()
        self._loggers.score(**metrics)

    def on_stop_training_error(self):
        self._callbacks.on_stop_training_error(self.status)

    def on_termination(self):
        self._callbacks.on_termination(self.status)
        self._loggers.terminate()
