import abc
import typing
import logging
import warnings
import numpy as np


class NanValueWarning(Warning):
    pass


def to_lower(string: str):
    return string.lower().replace(' ', '_')


def to_capital(string: str):
    return " ".join(word.capitalize() for word in to_lower(string).split("_"))


class AverageScore:
    """Computes and stores the average and current value"""
    __slots__ = ['_name', '_sum', '_count', '_value', '_fmtstr']

    def __init__(self, name: str, fmt: str = ":.4e", reset_on=None, by=1):
        assert isinstance(name, str), """name must be a string"""
        self._name = to_capital(name)
        self.set_format(fmt)
        self._value = np.nan
        self._sum = 0
        self._count = 0
        MetricHandler.register(self, reset_trigger=reset_on, n=by)

    @property
    def name(self) -> str:
        return to_lower(self._name)

    @property
    def counter(self) -> int:
        return self._count

    @property
    def average(self) -> float:
        return self._sum / self._count if self._count != 0 else np.nan

    @property
    def value(self) -> float:
        return self._value

    def set_format(self, fmt):
        self._fmtstr = f"{self._name}" + "={average" + fmt + "}"

    def reset(self) -> None:
        self._value = np.nan
        self._sum = 0
        self._count = 0

    def update(self, value, n=1) -> None:
        self._value = value
        self._sum += value * n
        self._count += n

    def __len__(self) -> int:
        return self._count

    def __str__(self) -> str:
        return self._fmtstr.format(average=self.average)

    def __repr__(self) -> str:
        return self._fmtstr.format(average=self.average)


class callbackmethod:
    """ a decorator for creating a callback list """
    __slots__ = ['_name', 'callbacks']

    def __init__(self, ep):
        self._name = ep.__qualname__
        self.callbacks: typing.List[typing.Tuple[AverageScore, int]] = list()

    def __call__(self, endpoint, n=1):
        assert isinstance(endpoint, AverageScore), endpoint
        self.callbacks.append((endpoint, n))
        return endpoint

    def trigger(self):
        for score, threshold in self.callbacks:
            if threshold <= score.counter:
                score.reset()


class MetricHandler:
    # neden class? registry icin
    # neden instance? sadece bi subseti hesaplamak icin
    __scores__: typing.Dict[str, AverageScore] = dict()
    __counts__: typing.Dict[str, int] = dict()

    def __init__(self, metrics=None) -> None:
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(15)

    @classmethod
    def register(cls, score, reset_trigger=None, n=1):
        assert isinstance(score, AverageScore)
        assert reset_trigger in ('step_end', 'epoch_end', None)
        cls.__scores__[score.name] = score

        if reset_trigger == 'step_end':
            cls.reset_on_step_end(score, n=n)
        if reset_trigger == 'epoch_end':
            cls.reset_on_epoch_end(score, n=n)

    def log_values(self, level, epoch_index, batch_index):
        score_values = self.score_values()
        score_values['epoch'] = epoch_index
        score_values['batch_index'] = batch_index
        self.logger.log(level, score_values)

    def log_averages(self, level, epoch_index, batch_index):
        score_values = self.score_averages()
        score_values['epoch'] = epoch_index
        score_values['batch_index'] = batch_index
        self.logger.log(level, score_values)

    @classmethod
    def get_score(self, score_name) -> float:
        return self.__scores__[score_name]

    @classmethod
    def score_values(self) -> typing.Dict[str, float]:
        return {name: score._value for name, score in self.__scores__.items()}

    @classmethod
    def score_averages(self) -> typing.Dict[str, float]:
        return {name: score.average for name, score in self.__scores__.items()}

    @callbackmethod
    def reset_on_step_end(self): pass

    @callbackmethod
    def reset_on_epoch_end(self): pass
