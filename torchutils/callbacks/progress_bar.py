from typing import List
from .base import TrainerCallback
from ..trainer.utils import (
    IterationArguments,
    IterationStatus,
    IterationInterface
)
from ..logging.pbar import (
    EpochProgressBar,
    BatchProgressBar,
    EvalProgressBar
)


class ProgressBar(TrainerCallback):
    """
    A callback which visualises the state of each training
    and evaluation epoch using a progress bar
    """

    def __init__(self, *score_names: str):
        self._test_bar = EvalProgressBar()
        self._epoch_bar = EpochProgressBar()
        self._step_bar = BatchProgressBar()
        self._score_names: List[str] = score_names
        self._args: IterationArguments = None

    def on_initialization(self, args: IterationArguments):
        self._args = args

    # Construction part
    def on_training_begin(self, stat: IterationStatus):
        self._epoch_bar.open(self._args)

    def on_training_epoch_begin(self, stat: IterationStatus):
        self._step_bar.open(self._args)

    def on_validation_run_begin(self, stat: IterationStatus):
        self._test_bar.open(self._args)

    def on_evaluation_run_begin(self, stat: IterationStatus):
        self._test_bar.open(self._args)

    # Incrementation part
    def on_training_step_end(self, batch: IterationInterface):
        scores = batch.get_current_scores(*self._score_names)
        self._step_bar.log_scores(scores)
        self._step_bar.update(1)

    def on_training_epoch_end(self, epoch: IterationInterface):
        scores = epoch.get_current_scores(*self._score_names)

        self._step_bar.log_scores(scores)
        self._epoch_bar.log_scores(scores)

        self._epoch_bar.update(1)
        self._step_bar.close()
        self._epoch_bar.update(0)

    def on_validation_step_end(self, batch: IterationInterface):
        scores = batch.get_current_scores(*self._score_names)
        self._test_bar.log_scores(scores)
        self._test_bar.update(1)
        self._epoch_bar.update(0)

    def on_evaluation_step_end(self, batch: IterationInterface):
        scores = batch.get_current_scores(*self._score_names)
        self._test_bar.log_scores(scores)
        self._test_bar.update(1)
        self._epoch_bar.update(0)

    # Destruction part
    def on_training_end(self, stat: IterationStatus):
        self._epoch_bar.close()

    def on_validation_run_end(self, epoch: IterationInterface):
        self._test_bar.close()
        self._epoch_bar.update(0)

    def on_evaluation_run_end(self, epoch: IterationInterface):
        self._test_bar.close()
        self._epoch_bar.update(0)
