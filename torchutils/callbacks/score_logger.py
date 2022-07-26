from .base import TrainerCallback
from ..logging import LoggerHandler, LoggingEvent
from ..metrics import MetricHandler
from ..trainer.utils import (
    IterationArguments,
    IterationStatus,
    IterationInterface
)


class ScoreLoggerCallback(TrainerCallback):
    """
    A callback which visualises the state of each training
    and evaluation epoch using a progress bar
    """

    def __init__(self, *score_names: str):
        super().__init__()
        self._handler: LoggerHandler = LoggerHandler.getHandler()
        self._args: IterationArguments = None
        if len(score_names) == 0:
            self._score_names = list(
                MetricHandler.MetricRegistrar.__score__.keys()
            )
            print(self._score_names)
        else:
            self._score_names = score_names

    def on_initialization(self, args: IterationArguments):
        self._args = args

    # Training
    def on_training_begin(self, status: IterationStatus):
        status.set_status_code(
            IterationStatus.StatusCode.STARTED
        )
        self._handler.initialize(args=self._args,
                                 event=LoggingEvent.TRAINING_EPOCH)

    def on_training_epoch_begin(self, stat: IterationStatus):
        self._handler.initialize(args=self._args,
                                 event=LoggingEvent.TRAINING_BATCH)

    def on_training_step_end(self, batch: IterationInterface):
        # self._log.event = LoggingEvent.TRAINING_BATCH
        # self._log.log_scores(batch.get_current_scores(*self._score_names))
        self._log.update(1)

    # Validation
    def on_validation_run_begin(self, stat: IterationStatus):
        self._handler.initialize(args=self._args,
                                 event=LoggingEvent.VALIDATION_RUN)

    def on_validation_step_end(self, batch: IterationInterface):
        # self._log.event = LoggingEvent.VALIDATION_RUN
        # @TODO: run at validation run end
        self._log.log_scores({
            f'valid_{score}': value for score,
            value in batch.get_current_scores(*self._score_names).items()
        })
        self._log.update(1)

    def on_validation_run_end(self, epoch: IterationInterface):
        self._handler.terminate(stats=epoch.status,
                                event=LoggingEvent.VALIDATION_RUN)

    def on_training_epoch_end(self, epoch: IterationInterface):
        self._handler.terminate(stats=epoch.status,
                                event=LoggingEvent.TRAINING_BATCH)
        # self._log.event = LoggingEvent.TRAINING_EPOCH
        self._log.log_scores(epoch.get_current_scores(*self._score_names))
        self._log.update(1)

    def on_training_end(self, stat: IterationStatus):
        self._handler.terminate(
            stats=stat,
            event=LoggingEvent.TRAINING_EPOCH
        )

    # Evaluation
    def on_evaluation_run_begin(self, stat: IterationStatus):
        self._handler.initialize(args=self._args,
                                 event=LoggingEvent.EVALUATION_RUN)

    def on_evaluation_step_end(self, batch: IterationInterface):
        # self._log.event = LoggingEvent.EVALUATION_RUN
        self._log.log_scores({
            f'eval_{score}': value for score,
            value in batch.get_current_scores(*self._score_names).items()
        })
        self._log.update(1)

    def on_evaluation_run_end(self, stat: IterationStatus):
        self._handler.terminate(stats=stat,
                                event=LoggingEvent.EVALUATION_RUN)

    def on_stop_training_error(self, stat: IterationStatus):
        self._handler.terminate(stats=stat, event=LoggingEvent.TRAINING_BATCH)
        self._handler.terminate(stats=stat, event=LoggingEvent.TRAINING_EPOCH)
        self._handler.terminate(stats=stat, event=LoggingEvent.EVALUATION_RUN)
        self._handler.terminate(stats=stat, event=LoggingEvent.VALIDATION_RUN)
