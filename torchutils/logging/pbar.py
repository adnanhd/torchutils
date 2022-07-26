import os
import warnings
import logging
from typing import Dict
from tqdm.autonotebook import tqdm
from .base import TrainerLogger
from ..trainer.status import IterationStatus
from ..trainer.arguments import TrainingArguments, EvaluatingArguments, Hyperparameter


class ProgressBarLogger(TrainerLogger):
    __slots__ = ['_pbar', '_log_dict_', 'config', '_logger_', '_is_training_']

    def __init__(self, **config):
        self._pbar: tqdm = None
        self.config = config
        self._log_dict_ = dict()
        self._is_training_: bool = None
        self._logger_ = logging.getLogger(__name__)

    def open(self, args: Hyperparameter):
        if isinstance(args, TrainingArguments):
            args: TrainingArguments
            self.config['unit'] = 'epoch'
            self.config['initial'] = args.resume_epochs
            self.config['desc'] = 'Training'
            self.config['total'] = args.num_epochs
            self._is_training_ = True
        elif isinstance(args, EvaluatingArguments):
            args: EvaluatingArguments
            self.config['unit'] = 'benchmark'
            self.config['initia'] = 0
            self.config['desc'] = 'Evaluating'
            self.config['total'] = args.eval_dl.num_steps
            self._is_training_ = False
        else:
            self._is_training_ = None

        self.config['position'] = 0
        self.config.setdefault("file", os.sys.stdout)
        self.config.setdefault("dynamic_ncols", True)
        self.config.setdefault("ascii", True)
        self.config.setdefault("colour", "GREEN")

        if self._pbar is None:
            self._pbar = tqdm(**self.config)
        else:
            warnings.warn(
                f"{self.__class__.__name__} is already opened", RuntimeWarning
            )

    def log_scores(self,
                   scores: Dict[str, float],
                   status: IterationStatus):
        self._logger_.debug(f'{self.__class__.__name__}.log_scores called')
        self._log_dict_.update(scores)
        self._pbar.set_postfix(self._log_dict_)

    def update(self, n, status: IterationStatus):
        self._logger_.debug(f'{self.__class__.__name__}.update called')
        self._log_dict_.clear()
        self._pbar.set_postfix(self._log_dict_)
        self._pbar.update(n=n)

    def close(self, status: IterationStatus):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        else:
            warnings.warn(
                f"{self.__class__.__name__} is already closed", RuntimeWarning
            )
