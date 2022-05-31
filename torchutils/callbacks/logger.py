from torchutils.callbacks.base import TrainerCallback
from torchutils.logging import NoneLogger, TrainerLogger
from typing import NewType
LOGGER_CLASS = NewType('LOGGER_CLASS', type(TrainerLogger))

class _LoggingBaseCallback(TrainerCallback):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """
    TEST_LOGGER_CLASS: LOGGER_CLASS = NoneLogger
    STEP_LOGGER_CLASS: LOGGER_CLASS = NoneLogger
    EPOCH_LOGGER_CLASS: LOGGER_CLASS = NoneLogger
    #__slots__ = ('_test_bar', '_epoch_bar', '_step_bar', '_step_size')

    def __init__(self):
        self._test_bar: TrainerLogger = self.TEST_LOGGER_CLASS()
        self._step_bar: TrainerLogger = self.STEP_LOGGER_CLASS()
        self._epoch_bar: TrainerLogger = self.EPOCH_LOGGER_CLASS()
        self._step_size: int = None

    def on_training_begin(self, num_epochs=None, batch_size=None, step_size=None, **kwargs):
        self._epoch_bar.open(num_epochs)
        self._step_size = step_size

    def on_training_epoch_begin(self, trainer, epoch=None, **kwargs):
        self._step_bar = StepProgressBarLogger(self._step_size)
        self._step_bar.open(epoch)

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
        self._test_bar.open()
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

