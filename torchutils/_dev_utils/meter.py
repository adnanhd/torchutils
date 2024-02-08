import pydantic
import logging
import typing
import math


class AverageMeter(pydantic.BaseModel):
    name: str
    _sum: float = pydantic.PrivateAttr(0)
    _count: float = pydantic.PrivateAttr(0)
    _value: float = pydantic.PrivateAttr(math.nan)
    #_logger: logging.Logger = pydantic.PrivateAttr()

    def __init__(self, **kwds):
        super().__init__(**kwds)
        #self._logger = logging.getLogger(self.__class__.__name__)
        AVERAGE_SCORES_DICT[self.name] = self

    @pydantic.field_validator('name')
    def name_validator(cls, value):
        if value in AVERAGE_SCORES_DICT:
            raise ValueError(f'{value} names must be unique')
        return value
    
    def __hash__(self):
        return self.name.__hash__()
        

    @property
    def counter(self) -> int:
        return self._count

    @property
    def average(self) -> float:
        return self._sum / self._count if self._count != 0 else math.nan

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = math.nan
        self._sum = 0
        self._count = 0

    def update(self, value) -> None:
        self._value = value
        self._sum += value
        self._count += 1
    

AVERAGE_SCORES_DICT: typing.Dict[str, AverageMeter] = dict()