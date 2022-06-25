import numpy as np
import warnings


class NanValueWarning(Warning):
    pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f"):
        assert isinstance(name, str), """name must be a string"""
        self._name = name
        self.fmt = fmt
        self.reset()

    @property
    def name(self):
        return self._name.lower().replace(' ', '_')

    @property
    def average(self):
        return self.sum / self.count if self.count != 0 else np.nan

    def reset(self) -> None:
        self.value = np.nan
        self.sum = 0
        self.count = 0

    def update(self, value, n=1) -> None:
        if np.isnan(value):
            warnings.warn(f"Metric({self._name}) receiveed a nan value)")
        self.value = value
        self.sum += value * n
        self.count += n

    def __str__(self) -> str:
        fmtstr = "{name} ({average" + self.fmt + "})"
        return fmtstr.format(name=self._name, average=self.average)

    def __repr__(self) -> str:
        return f"{self.name}={self.value}"
