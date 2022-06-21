from torchutils.callbacks.base import TrainerCallback
from torchutils.logging import LoggerHandler, LoggingEvent
from torchutils.utils.pydantic import (
    HandlerArguments,
    TrainerStatus,
    CurrentIterationStatus
)


class ScoreLoggerCallback(TrainerCallback):
    """
    A callback which visualises the state of each training
    and evaluation epoch using a progress bar
    """

    def __init__(self, *score_names: str):
        super().__init__()
        self._handler: LoggerHandler = LoggerHandler.getHandler()
        self._args: HandlerArguments = None

    def on_initialization(self, args: HandlerArguments):
        self._args = args

    # Training
    def on_training_begin(self, stat: TrainerStatus):
        self._handler.initialize(self._args, event=LoggingEvent.TRAINING_EPOCH)

    def on_training_epoch_begin(self, stat: TrainerStatus):
        self._handler.initialize(self._args, event=LoggingEvent.TRAINING_BATCH)

    def on_training_step_end(self, batch: CurrentIterationStatus):
        self._log.log_scores(
            msg=batch.get_current_scores(),
            event=LoggingEvent.TRAINING_BATCH
        )
        self._log.update(1, event=LoggingEvent.TRAINING_BATCH)

    def on_training_epoch_end(self, epoch: CurrentIterationStatus):
        self._handler.terminate(event=LoggingEvent.TRAINING_BATCH)
        self._log.log_scores(
            msg=epoch.get_current_scores(),
            event=LoggingEvent.TRAINING_EPOCH
        )
        self._log.update(1, event=LoggingEvent.TRAINING_EPOCH)

    def on_training_end(self, stat: TrainerStatus):
        self._handler.terminate(event=LoggingEvent.TRAINING_EPOCH)

    # Validation
    def on_validation_run_begin(self, stat: TrainerStatus):
        self._handler.initialize(
            self._args, event=LoggingEvent.VALIDATION_RUN)

    def on_validation_step_end(self, batch: CurrentIterationStatus):
        self._log.log_scores(
            msg=batch.get_current_scores(),
            event=LoggingEvent.VALIDATION_RUN
        )
        self._log.update(1, event=LoggingEvent.VALIDATION_RUN)

    def on_validation_run_end(self, stat: TrainerStatus):
        self._handler.terminate(event=LoggingEvent.VALIDATION_RUN)

    # Evaluation
    def on_evaluation_run_begin(self, stat: TrainerStatus):
        self._handler.initialize(
            self._args, event=LoggingEvent.EVALUATION_RUN)

    def on_evaluation_step_end(self, batch: CurrentIterationStatus):
        self._log.log_scores(
            msg=batch.get_current_scores(),
            event=LoggingEvent.EVALUATION_RUN
        )
        self._log.update(1, event=LoggingEvent.EVALUATION_RUN)

    def on_evaluation_run_end(self, stat: TrainerStatus):
        self._handler.terminate(event=LoggingEvent.EVALUATION_RUN)
