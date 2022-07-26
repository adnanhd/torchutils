from .base import TrainerCallback
import logging
from ..logging import LoggerInterface
from ..trainer.utils import (
    TrainingArguments,
    EvaluatingArguments,
    IterationInterface
)


class ScoreLoggerCallback(TrainerCallback):
    """
    A callback which visualises the state of each training
    and evaluation epoch using a progress bar
    """

    def __init__(self, *score_names: str):
        super().__init__()
        self._handler: LoggerInterface = None
        self._score_names = score_names
        self._logger = logging.getLogger(__name__)

    def on_initialization(self, handlers: LoggerInterface):
        self._handler = handlers

    def on_training_begin(self, hparams: TrainingArguments):
        # self._handler.log_hparams(hparams)
        pass

    def on_training_epoch_end(self, epoch: IterationInterface):
        self._handler.log_scores({
            f'train_{score}': value for score,
            value in epoch.get_current_scores(*self._score_names).items()
        })

    def on_validation_run_end(self, epoch: IterationInterface):
        # self._log.event = LoggingEvent.VALIDATION_RUN
        # @TODO: run at validation run end
        self._handler.log_scores({
            f'valid_{score}': value for score,
            value in epoch.get_current_scores(*self._score_names).items()
        })
