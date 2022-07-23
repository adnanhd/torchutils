# Import Handlers
from torchutils.metrics import MetricHandler, AverageMeter
from torchutils.logging import LoggerHandler, TrainerLogger, LoggingEvent, ExperimentProfiler
from torchutils.callbacks import CallbackHandler, TrainerCallback

import typing
import logging

# Import Arguments
from ..metrics.history import DataFrameRunHistory
from .utils import (
    TrainingArguments,
    EvaluatingArguments,
    IterationArguments,
    IterationInterface,
    IterationStatus
)


class IterationHandler(object):
    __slots__ = [
        'logger',
        'interface',
        'hparams',
        '_callbacks',
        '_loggers',
        '_metrics',
        '_history',
    ]

    def __init__(self,
                 metrics: MetricHandler,
                 loggers: LoggerHandler,
                 callbacks: CallbackHandler,
                 history: typing.Set[str] = set()):
        self.logger = logging.getLogger()
        self._metrics = metrics
        self._loggers = loggers
        self._callbacks = callbacks
        self.hparams: IterationArguments
        # @TODO: current_epoch=self.hparams.resume_epochs
        self._history = DataFrameRunHistory(history)
        self.interface: IterationInterface

    @property
    def status(self) -> IterationStatus:
        return self.interface.status

    # @property
    # def hparams(self) -> IterationArguments:
    #     return self.interface.hparams

    def on_initialization(self):
        self._metrics.reset_score_values()
        # self.profiler.set_status(self.interface.status)
        self._callbacks.on_initialization(self.interface.hparams)

    def on_stop_training_error(self):
        self._callbacks.on_stop_training_error(self.interface.status)

    def on_termination(self):
        self._callbacks.on_termination(self.interface.status)


class TrainingHandler(IterationHandler):
    def __init__(self, arguments: TrainingArguments, **kwargs):
        super().__init__(**kwargs)
        self.hparams = arguments
        self.interface = IterationInterface(
            metrics=self._metrics,
            history=self._history,
            hparams=self.hparams
        )

    def on_training_begin(self):
        self._callbacks.on_training_begin(self.interface.status)

    def on_training_epoch_begin(self, epoch_idx):
        self.interface.status.current_epoch = epoch_idx
        self._callbacks.on_training_epoch_begin(self.interface.status)

    def on_training_step_begin(self, batch_idx):
        self.interface.status.current_batch = batch_idx
        self._callbacks.on_training_step_begin(self.interface.status)

    def on_training_step_end(self, x, y, y_pred):
        # @TODO: call collate_fn with argument batch_idx
        # directly at DataLoader using pytorch API
        self.interface.collate_fn(input=x,
                                  preds=y_pred,
                                  target=y)
        # @TODO: place event in this instance insead of loggers
        self._callbacks.on_training_step_end(self.interface)

    def on_training_epoch_end(self):
        # @TODO: call from _history
        self.interface.set_metric_scores()
        self._callbacks.on_training_epoch_end(self.interface)
        self.interface.reset_metric_scores()

    def on_training_end(self):
        self._callbacks.on_training_end(self.interface.status)

    def on_validation_run_begin(self):
        self._callbacks.on_validation_run_begin(self.interface.status)

    def on_validation_step_begin(self, batch_idx=-1):
        self.interface.status.current_batch = batch_idx
        self._callbacks.on_validation_step_begin(self.interface.status)

    def on_validation_step_end(self, x, y, y_pred):
        self.interface.collate_fn(input=x,
                                  preds=y_pred,
                                  target=y)
        self._callbacks.on_validation_step_end(self.interface)

    def on_validation_run_end(self):
        self.interface.set_metric_scores()
        self._callbacks.on_validation_run_end(self.interface)
        self.interface.reset_metric_scores()


class EvaluatingHandler(IterationHandler):
    def __init__(self, arguments: EvaluatingArguments, **kwargs):
        super().__init__(**kwargs)
        self.hparams = arguments
        self.interface = IterationInterface(
            metrics=self._metrics,
            history=self._history,
            hparams=self.hparams
        )

    def on_evaluation_run_begin(self):
        self._callbacks.on_evaluation_run_begin(self.interface.status)

    def on_evaluation_step_begin(self):
        self._callbacks.on_evaluation_step_begin(self.interface.status)

    def on_evaluation_step_end(self, x, y, y_pred):
        self.interface.collate_fn(input=x,
                                  preds=y_pred,
                                  target=y)
        self._loggers.set_event(LoggingEvent.EVALUATION_RUN)
        self._callbacks.on_evaluation_step_end(self.interface)

    def on_evaluation_run_end(self):
        self.interface.set_metric_scores()
        self._callbacks.on_evaluation_run_end(self.interface)
        self.interface.reset_metric_scores()
