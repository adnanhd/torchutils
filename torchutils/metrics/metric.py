import typing
import logging
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

    def __init__(self, name: str, fmt: str = ":.4e", reset_on=None):
        assert isinstance(name, str), """name must be a string"""
        self._name = to_capital(name)
        self.set_format(fmt)
        self._value = np.nan
        self._sum = 0
        self._count = 0
        assert reset_on in ('epoch_end', None)
        MetricHandler.register(self, reset_trigger=reset_on, n=1)

    def set_format(self, fmt):
        self._fmtstr = f"{self._name}" + "={average" + fmt + "}"

    @property
    def format(self):
        return self._fmtstr

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
    __functs__: typing.Dict[str, typing.Callable] = dict()

    def __init__(self, metrics: typing.Set[str] = tuple()) -> None:
        self.metrics = metrics.union({'batch_index', 'epoch'})
        self.scores = dict()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        self.logger.setLevel(15)

    def add_handlers(self, handlers: typing.List[logging.Handler] = list()):
        for hdlr in handlers:
            self.logger.addHandler(hdlr)

    def remove_handlers(self, handlers: typing.List[logging.Handler] = list()):
        for hdlr in handlers:
            self.logger.removeHandler(hdlr)

    @classmethod
    def register(cls, score, reset_trigger=None, n=1):
        assert isinstance(score, AverageScore)
        assert reset_trigger in ('step_end', 'epoch_end', None)
        cls.__scores__[score.name] = score

        if reset_trigger == 'step_end':
            cls.reset_scores_on_step_end(score, n=n)
        elif reset_trigger == 'epoch_end':
            cls.reset_scores_on_epoch_end(score, n=n)

    def log(self, level, epoch_index, batch_index):
        self.scores['epoch'] = epoch_index
        self.scores['batch_index'] = batch_index
        scores = map(self.scores.__getitem__, self.metrics)
        self.logger.log(level, dict(zip(self.metrics, scores)))

    def score_dict(self):
        return self.scores

    def save_values_to_score_dict(self):
        for name, score in self.__scores__.items():
            self.scores[name] = score.value

    def save_averages_to_score_dict(self):
        for name, score in self.__scores__.items():
            self.scores[name] = score.average

    @callbackmethod
    def reset_scores_on_step_end(self): pass

    @callbackmethod
    def reset_scores_on_epoch_end(self): pass
