from torchutils.trainer.utils import TrainingArguments, IterationStatus
from torchutils.logging import ScoreLogger
from tqdm.autonotebook import tqdm
from typing import Dict
import warnings
import os


class ProgressBarLogger(ScoreLogger):
    __slots__ = ['_pbar', '_log_dict_', 'config']

    def __init__(self, **config):
        self._pbar = None
        self.config = config
        self._log_dict_ = dict()

    def open(self, args: TrainingArguments = None):
        if self._pbar is None:
            self._pbar = tqdm(**self.config)
        else:
            warnings.warn(
                f"{self.__class__.__name__} is already opened", RuntimeWarning
            )

    def log_scores(self,
                   scores: Dict[str, float],
                   status: IterationStatus):
        self._log_dict_.update(scores)

    def update(self, n, status: IterationStatus):
        self._pbar.set_postfix(self._log_dict_)
        self._pbar.update(n=n)

    def close(self, status: IterationStatus):
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

    def open(self, args: TrainingArguments):
        self.config['initial'] = args.resume_epochs
        self.config['total'] = args.num_epochs
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

    def open(self, hparams: TrainingArguments, status: IterationStatus):
        self.config["desc"] = f"Epoch {status.current_epoch}"
        self.config['total'] = hparams.train_dl.num_steps
        super().open()


class ValidProgressBar(ProgressBarLogger):
    def __init__(self, is_valid: bool = True, **config):
        config.setdefault("unit", "sample")
        config.setdefault("initial", 0)
        config.setdefault("position", 0)
        config.setdefault('leave', False)
        config.setdefault("file", os.sys.stdout)
        config.setdefault("dynamic_ncols", True)
        config.setdefault("desc", "Evaluating")
        config.setdefault("colour", "YELLOW")
        super(ValidProgressBar, self).__init__(**config)

    def open(self, args: TrainingArguments):
        self.config['total'] = args.valid_dl.num_steps
        super().open()


class EvalProgressBar(ProgressBarLogger):
    def __init__(self, is_valid: bool = True, **config):
        config.setdefault("unit", "sample")
        config.setdefault("initial", 0)
        config.setdefault("position", 0)
        config.setdefault('leave', False)
        config.setdefault("file", os.sys.stdout)
        config.setdefault("dynamic_ncols", True)
        config.setdefault("desc", "Evaluating")
        config.setdefault("colour", "YELLOW")
        super(EvalProgressBar, self).__init__(**config)

    def open(self, args: TrainingArguments):
        self.config['total'] = args.eval_dl.num_steps
        super().open()
