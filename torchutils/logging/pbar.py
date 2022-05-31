from typing import overload
from torchutils.logging.handler import TrainerLogger
from tqdm import tqdm
import time
import os


class ProgressBarLogger(TrainerLogger):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """

    def __init__(self, total=None, delay=0.0):
        super(ProgressBarLogger, self).__init__()
        self.pbar = None
        self._total = total
        self._delay = delay
        self._logs = dict()

    def open(self, total, **kwargs):
            self.pbar = tqdm(total=total, **kwargs)

    def log(self, trainer=None, **kwargs):
        self._logs.update(kwargs)
        self.pbar.set_postfix(self._logs)

    def update(self, n=1):
        self.pbar.update(n)

    def close(self):
        self.pbar.close()
        if self._delay > 0.0:
            time.sleep(self._delay)
    
    def __call__(self,n=1, **kwargs):
        self.log(**kwargs)
        self.update(n=n)

class EpochProgressBarLogger(ProgressBarLogger):
    def open(self):
        super().open(
            total=self._total,
            unit="epoch",
            initial=0,
            file=os.sys.stdout,
            dynamic_ncols=True,
            desc=f"Training",
            ascii=True,
            colour="GREEN",
        )

class StepProgressBarLogger(ProgressBarLogger):
    def open(self, epoch):
        super().open(
            total=self._total,
            unit=f"batch",
            file=os.sys.stdout,
            dynamic_ncols=True,
            desc=f"Epoch {epoch}",
            colour="GREEN",
        )

class TestProgressBarLogger(ProgressBarLogger):
    def open(self):
        super().open(
            total=self._total,
            unit="sample",
            file=os.sys.stdout,
            dynamic_ncols=True,
            desc=f"Testing",
            colour="GREEN",
        )
        
