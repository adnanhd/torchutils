# Import Handlers
from torchutils.metrics import MetricHandler
from torchutils.logging import LoggingHandler
from torchutils.callbacks import CallbackHandler

from typing import Union, Optional, List

# Import Arguments
from torchutils.utils.pydantic import (
    HandlerArguments,
    TrainerStatus,
    CurrentIterationStatus,
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
    __slots__ = ['iter_status',
                 'arguments', '_loggers',
                 '_metrics', '_callbacks',
                 '__args_setter__']

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
        self.iter_status = CurrentIterationStatus(handler=self._metrics)
        # TODO: explain well
        self.__args_setter__ = self.arguments.set_arguments()

    def set_arguments(self,
                      args: Union[TrainingArguments, EvaluatingArguments],
                      eval_dl: Optional[TrainerDataLoader] = None,
                      train_dl: Optional[TrainerDataLoader] = None,
                      valid_dl: Optional[TrainerDataLoader] = None,
                      ):
        dataloaders = {'train_dl': train_dl,
                       'valid_dl': valid_dl,
                       'eval_dl': eval_dl}
        dataloaders = {key: dataloader for key, dataloader
                       in dataloaders.items() if dataloader is not None}
        self.__args_setter__(args, dataloaders)

    @property
    def status(self):
        return self.arguments.status

    def compile(
            self,
            loggers=list(),
            metrics=list(),
            callbacks=list(),
    ):
        self._loggers.add_loggers(loggers)
        self._metrics._add_scores(metrics)
        self._callbacks.add_callbacks(callbacks)
        score_names = self._metrics.get_score_names()
        self.iter_status.set_score_names(*score_names)
        # self._valid_res.set_score_names(*score_names)

    def decompile(
            self,
            loggers=list(),
            metrics=list(),
            callbacks=list(),
    ):
        self._loggers.remove_logger(loggers)
        self._metrics._remove_scores(metrics)
        self._callbacks.remove_callbacks(callbacks)
        score_names = self._metrics.get_score_names()
        # TODO: remove *sign
        self.iter_status.set_score_names(*score_names)
        # self._valid_res.set_score_names(*score_names)

    def clear(self):
        self._loggers.clear_loggers()
        self._callbacks.clear_callbacks()
        self._metrics.clear_scores()
        self.iter_status.set_score_names()

    def on_initialization(self):
        self._loggers.initialize(self.arguments)
        self._callbacks.on_initialization(self.arguments)
        self.iter_status.reset_score_values()
        # self._valid_res.reset_score_values()

    def on_training_begin(self):
        self._loggers.model(self.arguments.model)
        self._callbacks.on_training_begin(self.status)

    def on_training_epoch_begin(self):
        self._callbacks.on_training_epoch_begin(self.status)

    def on_training_step_begin(self):
        self._callbacks.on_training_step_begin(self.status)

    def on_training_step_end(self, x, y, y_pred):
        self.iter_status.set_current_scores(x, y_true=y, y_pred=y_pred)
        scores = self._metrics.get_score_values()
        self._callbacks.on_training_step_end(self.iter_status)
        # TODO: remove this, apply it via callbacks
        self._loggers.score(**scores)

    def on_training_epoch_end(self):
        self.iter_status.average_scores()
        self._callbacks.on_training_epoch_end(self.iter_status)

    def on_training_end(self):
        self._callbacks.on_training_end(self.status)

    def on_validation_run_begin(self):
        self._callbacks.on_validation_run_begin(self.status)

    def on_validation_step_begin(self):
        self._callbacks.on_validation_step_begin(self.status)

    def on_validation_step_end(self, x, y, y_pred):
        self.iter_status.set_current_scores(x=x, y_true=y, y_pred=y_pred)
        self._callbacks.on_validation_step_end(self.iter_status)

    def on_validation_run_end(self):
        self.iter_status.average_scores()
        self._callbacks.on_validation_run_end(self.iter_status)

    def on_evaluation_run_begin(self):
        self._callbacks.on_evaluation_run_begin(self.status)

    def on_evaluation_step_begin(self):
        self._callbacks.on_evaluation_step_begin(self.status)

    def on_evaluation_step_end(self, x, y, y_pred):
        self.iter_status.set_current_scores(x=x, y_true=y, y_pred=y_pred)
        self._callbacks.on_evaluation_step_end(self.iter_status)

    def on_evaluation_run_end(self):
        self.iter_status.average_scores()
        self._callbacks.on_evaluation_run_end(self.iter_status)

    def on_stop_training_error(self):
        self._callbacks.on_stop_training_error(self.status)

    def on_termination(self):
        self._callbacks.on_termination(self.status)
        self._loggers.terminate()
