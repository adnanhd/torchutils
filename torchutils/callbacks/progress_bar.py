from torchutils.callbacks.base import TrainerCallback
from torchutils.logging.pbar import (
        TestProgressBarLogger, 
        EpochProgressBarLogger, 
        StepProgressBarLogger
)


class ProgressBar(TrainerCallback):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """
    def __init__(self):
        self._test_bar: TestProgressBarLogger = None
        self._epoch_bar: EpochProgressBarLogger = None
        self._step_bar: StepProgressBarLogger = None
        self._step_size: int = None

    def on_initialization(self):
        ...

    def on_training_begin(self, num_epochs=None, batch_size=None, step_size=None, **kwargs):
        self._epoch_bar = EpochProgressBarLogger(num_epochs)
        self._epoch_bar.get_logger()
        self._step_size = step_size

    def on_training_epoch_begin(self, trainer, epoch=None, **kwargs):
        self._step_bar = StepProgressBarLogger(self._step_size)
        self._step_bar.get_logger(epoch)

    def on_training_step_begin(self, trainer, **kwargs):
        ...

    def on_training_step_end(self,
                             trainer,
                             epoch=None,
                             batch=None,
                             batch_output=None,
                             **stepped_values):
        self._step_bar.log(**{k: v.item() for k, v in stepped_values.items()})
        self._step_bar.update()

    def on_training_epoch_end(self, trainer, epoch=None, **updated_values):
        self._step_bar.log(**{k: v.item() for k, v in updated_values.items()})
        self._step_bar.close()
        self._step_bar = None
        self._epoch_bar.log(**{k: v.item() for k, v in updated_values.items()})
        self._epoch_bar.update()

    def on_training_end(self, trainer, **kwargs):
        self._epoch_bar.close()
        self._epoch_bar = None

    def on_validation_run_begin(self, batch_size=None, step_size=None, **kwargs):
        self._test_bar = TestProgressBarLogger(step_size * batch_size)
        self._test_bar.get_logger()
        self._step_size = step_size

    def on_validation_step_end(self, 
                               trainer,
                               epoch=None,
                               batch=None,
                               batch_output=None,
                               **metric_values):
        self._test_bar.log(**{k: v.item() for k, v in metric_values.items()})
        self._test_bar.update(self._step_size)

    def on_validation_run_end(self, trainer, **kwargs):
        self._test_bar.close()
        self._test_bar = None

