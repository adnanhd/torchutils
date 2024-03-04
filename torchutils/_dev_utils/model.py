import typing
import logging
import inspect
import pydantic
from itertools import chain
from .meter import MeterBuffer, AverageMeter


class LoggerModel(pydantic.BaseModel):
    _logger: logging.Logger = pydantic.PrivateAttr()

    def __init__(self, level: int = logging.INFO):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(level=level)

    def __getattr__(self, name):
        logger: logging.Logger = self.__pydantic_private__['_logger']
        if hasattr(logger, name):
            return getattr(logger, name)
        return super().__getattr__(name)

    # Logger
    def log(self, *args, **kwds):
        self._logger.log(*args, **kwds)

    def log_debug(self, *args, **kwds):
        self._logger.debug(*args, **kwds)

    def log_info(self, *args, **kwds):
        self._logger.info(*args, **kwds)

    def log_warn(self, *args, **kwds):
        self._logger.warn(*args, **kwds)

    def log_error(self, *args, **kwds):
        self._logger.error(*args, **kwds)

    def log_fatal(self, *args, **kwds):
        self._logger.fatal(*args, **kwds)

    def add_handler(self, hdlr: logging.Handler):
        self._logger.addHandler(hdlr)

    def remove_handler(self, hdlr: logging.Handler):
        self._logger.removeHandler(hdlr)


class MeterModel(LoggerModel):
    _scores: MeterBuffer = pydantic.PrivateAttr()
    _buffer: typing.Dict[str, typing.Any] = pydantic.PrivateAttr(default_factory=dict)

    def __init__(self, scores: typing.Set[str] = set(), **kwds):
        assert isinstance(scores, set)
        super().__init__(**kwds)
        self._scores = MeterBuffer(scores=set(scores))

    # Score

    def register_score_name(self, name: str):
        self._scores.add_score_name(name=name)

    def register_score_meter(self, meter: AverageMeter):
        self._scores.add_score_meter(meter=meter)


class WriteMeterModel(MeterModel):

    def log_score(self, name: str, value: float, n: int = 1) -> None:
        self._scores.update_score_value(name=name, value=value, n=n)


class ReadMeterModel(MeterModel):

    def get_score_names(self) -> typing.Set[str]:
        return self._scores.get_score_names()

    def get_score_values(self) -> typing.Dict[str, float]:
        return {meter.name: meter.value for meter in self._scores.get_score_meters()}

    def get_score_averages(self) -> typing.Dict[str, float]:
        return {meter.name: meter.average for meter in self._scores.get_score_meters()}

    def get_score_meters(self) -> typing.Set[AverageMeter]:
        return self._scores.get_score_meters()



class MeterModelContainer(pydantic.BaseModel):
    """
    The :class:`CallbackHandler` is responsible for calling a list of callbacks.
    This class calls the callbacks in the order that they are given.
    """
    elements: typing.List[MeterModel]

    def __iter__(self):
        return self.elements.__iter__()

    def apply(self, fn_name):
        def method_call(*args, **kwds):
            return [
                getattr(el, fn_name)(*args, **kwds)
                for el in self.elements
                if hasattr(el, fn_name)
                and inspect.ismethod(getattr(el, fn_name))
            ]
        return method_call

    # Logger
    def add_handler(self, hdlr: logging.Handler):
        self.apply('add_handler')(hdlr)

    def remove_handler(self, hdlr: logging.Handler):
        self.apply('remove_handler')(hdlr)

    # Score
    def get_score_meters(self) -> typing.Set[AverageMeter]:
        return set(chain.from_iterable(self.apply('get_score_meters')()))

    def get_score_values(self) -> typing.Dict[str, float]:
        return dict(chain.from_iterable(map(dict.items, self.apply('get_score_values')())))

    def get_score_averages(self) -> typing.Dict[str, float]:
        return dict(chain.from_iterable(map(dict.items, self.apply('get_score_averages')())))

    def get_score_names(self) -> typing.Set[str]:
        return set(chain.from_iterable(self.apply('get_score_names')()))
