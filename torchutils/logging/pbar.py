from torchutils.utils.pydantic import HandlerArguments
from torchutils.logging import TrainerLogger, LoggingEvent
from collections import OrderedDict
from typing import Dict, Optional, List
from tqdm.autonotebook import tqdm
import os


class ProgressBarLogger(TrainerLogger):
    __slots__ = ['_pbar', '_logs', '_args']

    def __init__(self, **kwargs):
        self._logs = OrderedDict()
        self._args = kwargs.copy()
        self._pbar = None

    @classmethod
    def getEvent(cls) -> Dict[TrainerLogger, List[LoggingEvent]]:
        return {
            StepProgressBar(): [LoggingEvent.TRAINING_BATCH],
            EpochProgressBar(): [LoggingEvent.TRAINING_EPOCH],
            SampleProgressBar(): [LoggingEvent.EVALUATION_RUN,
                                  LoggingEvent.VALIDATION_RUN]
        }

    @classmethod
    def getLogger(cls, event: LoggingEvent) -> LoggingEvent:
        assert isinstance(event, LoggingEvent)
        if event in [LoggingEvent.TRAINING_BATCH]:
            return StepProgressBar()
        elif event in [LoggingEvent.TRAINING_EPOCH]:
            return EpochProgressBar()
        else:
            return SampleProgressBar()

    def open(self, total: int, args: HandlerArguments = None):
        if self._pbar is None:
            self._pbar = tqdm(total=total, **self._args)

    def log_scores(self,
                   scores: Dict[str, float],
                   step: Optional[int] = None):
        self._logs.update(scores)

    def update(self, n=0):
        self._pbar.set_postfix(self._logs)
        self._pbar.update(n=n)

    def close(self):
        if self._pbar is not None:
            self._pbar.close()
        self._pbar = None


class EpochProgressBar(ProgressBarLogger):
    def __init__(self, **kwargs):
        kwargs.setdefault("unit", "epoch")
        kwargs.setdefault("initial", 0)
        kwargs.setdefault("position", 1)
        kwargs.setdefault('leave', True)
        kwargs.setdefault("file", os.sys.stdout)
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("desc", "Training")
        kwargs.setdefault("ascii", True)
        kwargs.setdefault("colour", "GREEN")
        super().__init__(**kwargs)

    def open(self, args: HandlerArguments):
        super().open(total=args.hparams.num_epochs)


class StepProgressBar(ProgressBarLogger):
    def __init__(self, **kwargs):
        kwargs.setdefault("unit", "batch")
        kwargs.setdefault("initial", 0)
        kwargs.setdefault("position", 0)
        kwargs.setdefault('leave', False)
        kwargs.setdefault("file", os.sys.stdout)
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("colour", "CYAN")
        super().__init__(**kwargs)

    def open(self, args: HandlerArguments):
        epochs = args.status.current_epoch
        self._args["desc"] = f"Epoch {epochs}"
        super().open(total=args.train_dl.num_steps)


class SampleProgressBar(ProgressBarLogger):
    def __init__(self, **kwargs):
        kwargs.setdefault("unit", "sample")
        kwargs.setdefault("initial", 0)
        kwargs.setdefault("position", 0)
        kwargs.setdefault('leave', False)
        kwargs.setdefault("file", os.sys.stdout)
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("desc", "Evaluating")
        kwargs.setdefault("colour", "YELLOW")
        super().__init__(**kwargs)

    def open(self, args: HandlerArguments, valid: bool = True):
        if args.valid_dl is not None:
            total = args.valid_dl.num_steps
        else:
            total = args.eval_dl.num_steps
        super().open(total=total)
