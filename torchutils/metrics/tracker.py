import typing
import warnings
import numpy as np
from torchutils.utils.pydantic.types import NpTorchType
from .utils import has_allowed_arguments, to_capital, to_lower


class NanValueWarning(Warning):
    pass


class AverageMeter(object):
    """Computes and stores the average and current value"""
    __slots__ = ['_name', '_sum', '_count', 'value', 'fmt']

    def __init__(self, name: str, fmt: str = ":f"):
        assert isinstance(name, str), """name must be a string"""
        self._name = name  # to_capital(name)
        self.fmt = fmt
        self.reset()
        MetricRegistrar.register_meter(self)

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
            warnings.warn(f"Metric({self._name}) receiveed a nan value)")
        self.value = value
        self._sum += value * n
        self._count += n

    def __add__(self, value) -> "AverageMeter":
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


class AverageMeterFunction(AverageMeter):
    def __init__(self, name: str,
                 fn: typing.Callable[[NpTorchType, NpTorchType], NpTorchType]):
        assert callable(fn) and has_allowed_arguments(fn)
        assert isinstance(name, str)
        super().__init__(name=name)
        self.fn = fn

    def update(self, preds: NpTorchType, target: NpTorchType, n: int = 1) -> None:
        super().update(float(self.fn(preds=preds, target=target)), n=n)

    def __add__(self, value):
        ...

    def __repr__(self) -> str:
        fmtstr = "{func}({name}={average" + self.fmt + "})"
        return fmtstr.format(name=self.name, func=self.fn,
                             average=self.average)


class AverageMeterModule_Base(object):
    def __init__(self, *meters: str):
        assert all(isinstance(name, str) for name in meters)
        scores = {name: AverageMeter(name) for name in meters}
        super().__setattr__("__scores__", scores)

    def __setattr__(self, name, value):
        if name in super().__getattribute__("__scores__"):
            if not isinstance(value, AverageMeter) or value.name != name:
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


class MetricRegistrar(object):
    __functional__: typing.Dict[str, typing.Callable] = {
        # name.lower(): fn for name, fn in vars(f).items()
        # if callable(fn) and has_allowed_arguments(fn)
    }
    __score__: typing.Dict[str, AverageMeter] = {
    }

    @classmethod
    def register_meter(cls, meter: AverageMeter) -> None:
        assert isinstance(meter, AverageMeter)
        # if not name.istitle():
        # name = to_capital(name)
        if meter._name in cls.__score__:
            raise KeyError(
                f"{meter._name} is already registered."
            )
        else:
            cls.__score__.__setitem__(meter._name, meter)

    @classmethod
    def register_functional(cls, name: str, fn: typing.Callable) -> None:
        assert callable(fn) and has_allowed_arguments(fn)
        assert isinstance(name, str)
        # name = to_lower(name)
        if name in cls.__functional__:
            raise KeyError(
                f"{name} is already registered."
            )
        else:
            cls.__functional__.__setitem__(
                name, AverageMeterFunction(name, fn))

    @classmethod
    def unregister_meter(cls, name: str):
        # name = to_capital(name)
        cls.__score__.pop(name)

    @classmethod
    def clear_meters(cls):
        cls.__score__.clear()
        cls.__functional__.clear()
