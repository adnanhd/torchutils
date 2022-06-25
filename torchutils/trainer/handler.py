# Import Handlers
from torchutils.metrics import MetricHandler, AverageMeter
from torchutils.logging import LoggerHandler, TrainerLogger, LoggingEvent
from torchutils.callbacks import CallbackHandler, TrainerCallback

from typing import Optional
import typing

# Import Arguments
from torchutils.utils.pydantic import (
    HandlerArguments,
    TrainerStatus,
    CurrentIterationStatus,
    TrainerModel,
    TrainerDataLoader
)


class TrainerHandler():
    __slots__ = [
        'trainer_proxy',
        '_arguments',
        '_loggers',
        '_metrics',
        '_callbacks',
        '_log'
    ]

    def __init__(self):
        self._metrics = MetricHandler()
        self._callbacks = CallbackHandler()
        self._loggers: LoggerHandler = LoggerHandler.getHandler()
        self._log = LoggerHandler.getProxy()
        self._arguments: HandlerArguments = None
        self.trainer_proxy = CurrentIterationStatus(handler=self._metrics)

    @property
    def status(self) -> TrainerStatus:
        return self._arguments.status

    @property
    def hparams(self):
        return self._arguments.hparams

    def compile_model_and_hparams(self,
                                  model: TrainerModel,
                                  eval_dl: Optional[TrainerDataLoader] = None,
                                  train_dl: Optional[TrainerDataLoader] = None,
                                  valid_dl: Optional[TrainerDataLoader] = None,
                                  **hparams) -> None:
        self._arguments = HandlerArguments(
            model=model,
            train_dl=train_dl,
            valid_dl=valid_dl,
            eval_dl=eval_dl,
            **hparams
        )
        self.trainer_proxy.status = self._arguments.status
        self._loggers.setStatus(self._arguments.status)

    def compile_handlers(
            self,
            loggers: typing.Dict[TrainerLogger,
                                 typing.Iterable[LoggingEvent]] = dict(),
            metrics: typing.Iterable[AverageMeter] = list(),
            callbacks: typing.Iterable[TrainerCallback] = list(),
    ):
        for logger, events in loggers.items():
            for event in events:
                self._loggers.add_logger(event=event, logger=logger)
        self._metrics.add_score_meters(*metrics)
        self._callbacks.add_callbacks(callbacks)
        score_names = self._metrics.get_score_names()
        self.trainer_proxy.set_score_names(score_names)

    def decompile_handlers(
            self,
            loggers: typing.Dict[TrainerLogger,
                                 typing.Iterable[LoggingEvent]] = dict(),
            metrics: typing.Iterable[AverageMeter] = list(),
            callbacks: typing.Iterable[TrainerCallback] = list(),
    ):
        for logger, events in loggers.items():
            for event in events:
                self._loggers.remove_logger(event=event, logger=logger)
        self._metrics.remove_score_meters(*metrics)
        self._callbacks.remove_callbacks(callbacks)
        # TODO: remove *sign
        score_names = self._metrics.get_score_names()
        self.trainer_proxy.set_score_names(score_names)

    def clear_handlers(self):
        self._loggers.clear_loggers()
        self._callbacks.clear_callbacks()
        score_names = self._metrics.get_score_names()
        self._metrics.remove_score_meters(*score_names)
        self.trainer_proxy.reset_score_names()

    def on_initialization(self):
        self._callbacks.on_initialization(self._arguments)
        self._metrics.init_score_history()
        self._metrics.reset_score_values()

    def on_training_begin(self):
        self._callbacks.on_training_begin(self.status)

    def on_training_epoch_begin(self):
        self._callbacks.on_training_epoch_begin(self.status)

    def on_training_step_begin(self):
        self._callbacks.on_training_step_begin(self.status)

    def on_training_step_end(self, x, y, y_pred):
        self._log.event = LoggingEvent.TRAINING_BATCH
        self.trainer_proxy.set_current_scores(x, y_true=y, y_pred=y_pred)
        self._callbacks.on_training_step_end(self.trainer_proxy)

    def on_training_epoch_end(self):
        self._log.event = LoggingEvent.TRAINING_EPOCH
        self.trainer_proxy.average_current_scores()
        self._callbacks.on_training_epoch_end(self.trainer_proxy)

    def on_training_end(self):
        self._callbacks.on_training_end(self.status)

    def on_validation_run_begin(self):
        self.trainer_proxy.average_current_scores()
        self._callbacks.on_validation_run_begin(self.status)

    def on_validation_step_begin(self):
        self._callbacks.on_validation_step_begin(self.status)

    def on_validation_step_end(self, x, y, y_pred):
        self._log.event = LoggingEvent.VALIDATION_RUN
        self.trainer_proxy.set_current_scores(x=x, y_true=y, y_pred=y_pred)
        self._callbacks.on_validation_step_end(self.trainer_proxy)

    def on_validation_run_end(self):
        self._callbacks.on_validation_run_end(self.trainer_proxy)

    def on_evaluation_run_begin(self):
        self._callbacks.on_evaluation_run_begin(self.status)

    def on_evaluation_step_begin(self):
        self._callbacks.on_evaluation_step_begin(self.status)

    def on_evaluation_step_end(self, x, y, y_pred):
        self._log.event = LoggingEvent.EVALUATION_RUN
        self.trainer_proxy.set_current_scores(x=x, y_true=y, y_pred=y_pred)
        self._callbacks.on_evaluation_step_end(self.trainer_proxy)

    def on_evaluation_run_end(self):
        self.trainer_proxy.average_current_scores()
        self._callbacks.on_evaluation_run_end(self.trainer_proxy)

    def on_stop_training_error(self):
        self._callbacks.on_stop_training_error(self.status)

    def on_termination(self):
        self._callbacks.on_termination(self.status)
