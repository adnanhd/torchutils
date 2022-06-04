from torchutils.callbacks.base import TrainerCallback
from torchutils.utils.pydantic import HandlerArguments, TrainerStatus, EpochResults, StepResults
from torchutils.logging.pbar import (
        EpochProgressBar,
        StepProgressBar,
        SampleProgressBar
)


class ProgressBar(TrainerCallback):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """
    def __init__(self):
        self._test_bar = SampleProgressBar()
        self._epoch_bar = EpochProgressBar()
        self._step_bar = StepProgressBar()
        self._args = None

    def on_initialization(self, args: HandlerArguments):
        self._args = args

    # Construction part
    def on_training_begin(self, stat: TrainerStatus):
        self._epoch_bar.open(self._args)

    def on_training_epoch_begin(self, stat: TrainerStatus):
        self._step_bar.open(self._args)

    def on_validation_run_begin(self, stat: TrainerStatus):
        self._test_bar.open(self._args)

    def on_evaluation_run_begin(self, stat: TrainerStatus):
        self._test_bar.open(self._args)

    # Incrementation part
    def on_training_epoch_end(self, epoch: EpochResults):
        self._step_bar.close()
        self._epoch_bar.update(1)

    def on_training_step_end(self, batch: StepResults):
        self._step_bar.update(1)

    def on_validation_step_end(self, batch: StepResults):
        self._test_bar.update(1)

    def on_evaluation_step_end(self, batch: StepResults):
        self._test_bar.update(1)

    # Destruction part
    def on_training_end(self, stat: TrainerStatus):
        self._step_bar.close()
        self._epoch_bar.close()

    def on_validation_run_end(self, epoch: EpochResults):
        self._test_bar.close()

    def on_evaluation_run_end(self, epoch: EpochResults):
        self._test_bar.close()

