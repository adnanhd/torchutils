from torchutils.utils.pydantic import HandlerArguments
from torchutils.logging import TrainerLogger
from collections import OrderedDict
from tqdm import tqdm
import time
import os

class ProgressBarLogger(TrainerLogger):
    __slots__ = ['_pbar', '_logs', '_args']
    def __init__(self, **kwargs):
        self._logs = OrderedDict()
        self._args = kwargs.copy()

    def open(self, total: int, args: HandlerArguments = None):
        self._pbar = tqdm(total=total, **self._args)

    def log_score(self, **kwargs):
        self._logs.update(kwargs)

    def update(self, n=0):
        self._pbar.set_postfix(self._logs)
        self._pbar.update(n=n)

    def close(self):
        self._pbar.close()


class EpochProgressBar(ProgressBarLogger):
    def __init__(self, **kwargs):
        kwargs.setdefault("unit", "epoch")
        kwargs.setdefault("initial", 0)
        kwargs.setdefault("file", os.sys.stdout)
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("desc", "Training")
        kwargs.setdefault("ascii", True)
        kwargs.setdefault("colour", "GREEN")
        super().__init__(**kwargs)

    def open(self, args: HandlerArguments):
        super().open(total=args.args.num_epochs)

    def _flush_epoch(self):
        self.update(1)

class StepProgressBar(ProgressBarLogger):
    def __init__(self, **kwargs):
        kwargs.setdefault("unit", "batch")
        kwargs.setdefault("initial", 0)
        kwargs.setdefault("file", os.sys.stdout)
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("colour", "CYAN")
        super().__init__(**kwargs)
    
    def open(self, args: HandlerArguments):
        epochs = args.status.current_epoch
        self._args["desc"] = f"Epoch {epochs}"
        super().open(total=args.train_dl.num_steps)

    def _flush_step(self):
        self.update(1)


class SampleProgressBar(ProgressBarLogger):
    def __init__(self, **kwargs):
        kwargs.setdefault("unit", "sample")
        kwargs.setdefault("initial", 0)
        kwargs.setdefault("file", os.sys.stdout)
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("desc", f"Evaluating")
        kwargs.setdefault("colour", "GREEN")
        super().__init__(**kwargs)

    def open(self, args: HandlerArguments):
        super().open(total=args.eval_dl.num_steps)

    def _flush_step(self):
        self.update(1)

