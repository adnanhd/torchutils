from ..utils import RegisteredBaseModel
import pydantic
import typing
import math


class AverageMeter(RegisteredBaseModel):
    _sum: float = pydantic.PrivateAttr(0)
    _count: float = pydantic.PrivateAttr(0)
    _value: float = pydantic.PrivateAttr(math.nan)
    @pydantic.computed_field
    def average(self) -> float:
        return float(self._sum / self._count) if self._count != 0 else math.nan
        
    @pydantic.computed_field
    def count(self) -> int:
        return int(self._count)
        
    @property
    def counter(self) -> int:
        import warnings
        warnings.warn('depricated')
        return self._count

    @property
    def value(self) -> float:
        return float(self._value)

    def update(self, value: float, n: int = 1) -> None:
        assert n > 0
        self._value = value
        self._sum += value
        self._count += n

    def reset(self) -> None:
        self._value = math.nan
        self._sum = 0
        self._count = 0


class _Base_MeterDict(pydantic.BaseModel):
    _scores: typing.Dict[str, AverageMeter] = pydantic.PrivateAttr(default_factory=dict)
    def __init__(self, scores: typing.Set[str]):
        super().__init__()
        scores = tuple(map(AverageMeter.get_instance, scores))
        names = map(lambda score: score.name, scores)
        self._scores = dict(zip(names, scores))
    
    @pydantic.computed_field
    def score_names(self) -> typing.Set[str]:
        return set(self._scores.keys())

    def get_score_names(self) -> typing.Set[str]:
        return set(self._scores.keys())
    
    def get_score_meters(self) -> typing.Set[AverageMeter]:
        return set(self._scores.values())
    
    def add_score_meter(self, meter: AverageMeter):
        assert meter.name not in self._scores.keys()
        self._scores[meter.name] = meter
    
    def add_score_name(self, name: str):
        assert name not in self._scores.keys()
        return self._scores.setdefault(name, AverageMeter(name=name))
    
    def pop_score_name(self, name: str) -> AverageMeter:
        return self._scores.pop(name) 


class MeterBuffer(_Base_MeterDict):
    def update_score_value(self, name: str, value: float, n: int):
        self._scores[name].update(value=value, n=n)

    def obtain_score_value(self, name: str) -> float:
        return self._scores[name].value

    def obtain_score_average(self, name: str) -> float:
        return self._scores[name].average

    def obtain_score_count(self, name: str) -> int:
        return self._scores[name].count


class MeterHandler(_Base_MeterDict):
    def __init__(self):
        from ..utils.interfaces._base import _REGISTRY
        super().__init__(scores=set(_REGISTRY[AverageMeter.__name__].keys()))

    def reset_meters(self):
        for meter in self.get_score_meters():
            meter.reset()
