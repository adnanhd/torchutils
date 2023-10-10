import abc
import typing
import warnings
import numpy as np


class NanValueWarning(Warning):
    pass

from .utils import eventtrigger


class MetricHandler:
    __scores__: typing.Dict[str, typing.Callable[[],float]] = dict()

    @classmethod
    def register(cls, score, reset='never'):
        assert isinstance(score, AverageScore)
        assert reset in ('step_end', 'epoch_end', 'never')
        if reset == 'step_end':
            cls.on_step_end_reset(score.reset)
        if reset == 'epoch_end':
            cls.on_epoch_end_reset(score.reset)
        cls.__scores__[score.name] = score

    @classmethod
    def get_score(self, score_name) -> float:
        return self.__scores__[score_name]
    
    @classmethod
    def score_values(self) -> typing.Dict[str, float]:
        return {name: score._value for name, score in self.__scores__.items()}
    
    @classmethod
    def score_averages(self) -> typing.Dict[str, float]:
        return {name: score.average for name, score in self.__scores__.items()}
    
    @eventtrigger
    def on_step_end_reset(self): pass

    @eventtrigger
    def on_epoch_end_reset(self): pass


def to_lower(string: str):
    return string.lower().replace(' ', '_')

def to_capital(string: str):
    return " ".join(word.capitalize() for word in to_lower(string).split("_"))


class AverageScore:
    """Computes and stores the average and current value"""
    __slots__ = ['_name', '_sum', '_count', '_value', '_fmtstr']

    def __init__(self, name: str, fmt: str = ":.4e"):
        assert isinstance(name, str), """name must be a string"""
        self._name = to_capital(name)
        self.set_format(fmt)
        self.reset()
        MetricHandler.register(self)

    @property
    def name(self) -> str:
        return to_lower(self._name)
    
    def set_format(self, fmt):
        self._fmtstr = f"{self._name}" + "={average" + fmt + "}"

    @property
    def total(self) -> int:
        return self._count

    @property
    def average(self) -> float:
        return self._sum / self._count if self._count != 0 else np.nan
    
    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = np.nan
        self._sum = 0
        self._count = 0

    def update(self, value, n=1) -> None:
        self._value = value
        self._sum += value * n
        self._count += n

    #def __add__(self, _value) -> "AverageScore":
    #    self.update(_value)
    #    return self

    def __len__(self) -> int:
        return self._count

    def __str__(self) -> str:
        return self._fmtstr.format(average=self.average)

    def __repr__(self) -> str:
        return self._fmtstr.format(average=self.average)