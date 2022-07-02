from torchutils.trainer.utils import HandlerArguments, TrainerStatus
from torchutils.logging import TrainerLogger, LoggingEvent
from tqdm.autonotebook import tqdm
from typing import Dict, List
import warnings
import os


class ProgressBarLogger(TrainerLogger):
    __slots__ = ['_pbar', '_log_dict_', 'config']

    def __init__(self, **config):
        self._pbar = None
        self.config = config
        self._log_dict_ = dict()

    @classmethod
    def getEvent(cls) -> Dict[TrainerLogger, List[LoggingEvent]]:
        return {
            BatchProgressBar(): [LoggingEvent.TRAINING_BATCH],
            EpochProgressBar(): [LoggingEvent.TRAINING_EPOCH],
            SampleProgressBar(): [LoggingEvent.EVALUATION_RUN,
                                  LoggingEvent.VALIDATION_RUN]
        }

    @classmethod
    def getLogger(cls, event: LoggingEvent, **kwargs) -> LoggingEvent:
        assert isinstance(event, LoggingEvent)
        if event == LoggingEvent.TRAINING_BATCH:
            return BatchProgressBar(**kwargs)
        elif event == LoggingEvent.TRAINING_EPOCH:
            return EpochProgressBar(**kwargs)
        elif event == LoggingEvent.VALIDATION_RUN:
            return SampleProgressBar(is_valid=True, **kwargs)
        else:
            return SampleProgressBar(is_valid=False, **kwargs)

    def open(self, args: HandlerArguments = None):
        if self._pbar is None:
            self._pbar = tqdm(**self.config)
        else:
            warnings.warn(
                f"{self.__class__.__name__} is already opened", RuntimeWarning
            )

    def log_scores(self,
                   scores: Dict[str, float],
                   status: TrainerStatus):
        self._log_dict_.update(scores)

    def update(self, n, status: TrainerStatus):
        self._pbar.set_postfix(self._log_dict_)
        self._pbar.update(n=n)

    def close(self, status: TrainerStatus):
        if self._pbar is not None:
            self._pbar.close()
        else:
            warnings.warn(
                f"{self.__class__.__name__} is already closed", RuntimeWarning
            )
        self._pbar = None


class EpochProgressBar(ProgressBarLogger):
    def __init__(self, **config):
        config.setdefault("unit", "epoch")
        config.setdefault("initial", 0)
        config.setdefault("position", 1)
        config.setdefault('leave', True)
        config.setdefault("file", os.sys.stdout)
        config.setdefault("dynamic_ncols", True)
        config.setdefault("desc", "Training")
        config.setdefault("ascii", True)
        config.setdefault("colour", "GREEN")
        super(EpochProgressBar, self).__init__(**config)

    def open(self, args: HandlerArguments):
        self.config['initial'] = args.hparams.resume_epochs
        self.config['total'] = args.hparams.num_epochs
        super().open()


class BatchProgressBar(ProgressBarLogger):
    def __init__(self, **config):
        config.setdefault("unit", "batch")
        config.setdefault("initial", 0)
        config.setdefault("position", 0)
        config.setdefault('leave', False)
        config.setdefault("file", os.sys.stdout)
        config.setdefault("dynamic_ncols", True)
        config.setdefault("colour", "CYAN")
        super(BatchProgressBar, self).__init__(**config)

    def open(self, args: HandlerArguments):
        epochs = args.status.current_epoch
        self.config["desc"] = f"Epoch {epochs}"
        self.config['total'] = args.train_dl.num_steps
        super().open()


class SampleProgressBar(ProgressBarLogger):
    __slots__ = ['__is_valid__']

    def __init__(self, is_valid: bool = True, **config):
        config.setdefault("unit", "sample")
        config.setdefault("initial", 0)
        config.setdefault("position", 0)
        config.setdefault('leave', False)
        config.setdefault("file", os.sys.stdout)
        config.setdefault("dynamic_ncols", True)
        config.setdefault("desc", "Evaluating")
        config.setdefault("colour", "YELLOW")
        super(SampleProgressBar, self).__init__(**config)
        self.__is_valid__ = is_valid

    def open(self, args: HandlerArguments):
        if self.__is_valid__:
            self.config['total'] = args.valid_dl.num_steps
        else:
            self.config['total'] = args.eval_dl.num_steps
        super().open()
