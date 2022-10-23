import abc
import typing
import warnings
import numpy as np
from torchutils.utils.pydantic.types import NpTorchType
from .utils import has_allowed_arguments, to_capital, to_lower


class NanValueWarning(Warning):
    pass


class AverageScore(object):
    """Computes and stores the average and current value"""
    __slots__ = ['_name', '_sum', '_count', 'value', 'fmt']

    def __init__(self, name: str, fmt: str = ":f"):
        assert isinstance(name, str), """name must be a string"""
        self._name = name  # to_capital(name)
        self.fmt = fmt
        self.reset()
        MetricRegistrar.register_score(self)

    @property
    def name(self) -> str:
        return to_lower(self._name)

    @property
    def average(self) -> float:
        return self._sum / self._count if self._count != 0 else np.nan

    @property
    def total(self) -> float:
        return self._count

    def reset(self) -> None:
        self.value = np.nan
        self._sum = 0
        self._count = 0

    def update(self, value, n=1) -> None:
        if np.isnan(value):
            if self._name == 'ld_score':
                import pdb
                pdb.set_trace()
            warnings.warn(f"Metric({self._name}) received NaN value)")
        self.value = value
        self._sum += value * n
        self._count += n

    def __add__(self, value) -> "AverageScore":
        self.update(value)
        return self

    def __len__(self) -> int:
        return self._count

    def __str__(self) -> str:
        fmtstr = "{name}={average" + self.fmt + "}"
        return fmtstr.format(name=self.name, average=self.average)

    def __repr__(self) -> str:
        fmtstr = "<{name}>({average" + self.fmt + "})"
        return fmtstr.format(name=to_capital(self._name), average=self.average)


class AverageScoreFunctional(AverageScore):
    def __init__(self, name: str,
                 fn: typing.Callable[[NpTorchType, NpTorchType], NpTorchType],
                 **hparams):
        assert callable(fn) and has_allowed_arguments(fn)
        assert isinstance(name, str)
        super().__init__(name=name)
        self.fn = fn
        self.kwargs = hparams

    def update(self, preds: NpTorchType, target: NpTorchType, n: int = 1) -> None:
        super().update(float(self.fn(preds=preds, target=target, **self.kwargs)), n=n)

    def __add__(self, value):
        ...

    def __repr__(self) -> str:
        fmtstr = "{func}({name}={average" + self.fmt + "})"
        return fmtstr.format(name=self.name, func=self.fn,
                             average=self.average)


class AverageScoreModule(AverageScore):
    def __init__(self, *names: str):
        assert all(isinstance(name, str) for name in names)
        scores = {name: AverageScore(name) for name in names}
        super().__setattr__("__scores__", scores)

    @property
    def _names(self) -> typing.Set[str]:
        return set(super().__getattribute__('__scores__').keys())

    @property
    def _name(self) -> str:
        return self.__class__.__name__

    def __setattr__(self, name, value):
        if name in super().__getattribute__("__scores__"):
            if not isinstance(value, AverageScore) or value.name != name:
                raise AttributeError(
                    f"Cannot set the initialized meter {name} to {value}"
                )
        else:
            super().__setattr__(name, value)

    def __getattribute__(self, name):
        __scores__ = super().__getattribute__('__scores__')
        if name in __scores__:
            return __scores__[name]
        else:
            return super().__getattribute__(name)

    @typing.overload
    def update(self, preds: NpTorchType, target: NpTorchType) -> NpTorchType:
        pass

    def __str__(self) -> str:
        return ", ".join(str(score) for score in super(
        ).__getattribute__('__scores__').values())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__str__()})"


FunctionalOrModule = typing.Union[AverageScoreFunctional, AverageScoreModule]


class MetricRegistrar(object):
    __metrics__: typing.Dict[str, FunctionalOrModule] = dict()
    # name.lower(): fn for name, fn in vars(f).items()
    # if callable(fn) and has_allowed_arguments(fn)
    __score__: typing.Dict[str, AverageScore] = dict()
    __score_to_metric__: typing.Dict[str, str] = dict()

    @classmethod
    def register_score(cls, meter: AverageScore) -> None:
        assert isinstance(meter, AverageScore)
        # if not name.istitle():
        # name = to_capital(name)
        if meter._name in cls.__score__:
            raise KeyError(
                f"{meter._name} is already registered."
            )
        else:
            cls.__score__.__setitem__(meter._name, meter)

    @classmethod
    def register_functional(cls, meter_fn: FunctionalOrModule) -> None:
        assert isinstance(meter_fn, AverageScoreFunctional) \
            or isinstance(meter_fn, AverageScoreModule)
        # name = to_lower(name)
        if meter_fn._name in cls.__metrics__:
            raise KeyError(
                f"{meter_fn._name} is already registered."
            )
        cls.__metrics__.__setitem__(meter_fn._name, meter_fn)
        if isinstance(meter_fn, AverageScoreFunctional):
            cls.__score_to_metric__[meter_fn._name] = meter_fn._name
        elif isinstance(meter_fn, AverageScoreModule):
            for score_name in meter_fn._names:
                cls.__score_to_metric__[score_name] = meter_fn._name

    @classmethod
    def unregister_score(cls, name: str) -> AverageScore:
        # name = to_capital(name)
        if name not in cls.__score__:
            raise KeyError(
                f"{name} is not registered."
            )
        else:
            # @TODO: pop or __delitem__
            return cls.__score__.pop(name)

    @classmethod
    def unregister_functional(cls, name) -> FunctionalOrModule:
        # name = to_lower(name)
        if name not in cls.__metrics__:
            raise KeyError(
                f"{name} is not registered."
            )
        else:
            cls.__score_to_metric__.pop(name)
            return cls.__metrics__.pop(name)

    @classmethod
    def clear(cls):
        cls.__score__.clear()
        cls.__metrics__.clear()
        cls.__score_to_metric__.clear()
