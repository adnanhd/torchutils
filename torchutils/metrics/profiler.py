import pydantic
import logging
import typing
import math
import time


class Profiler(pydantic.BaseModel):
    name: str
    _start_time: float = pydantic.PrivateAttr(math.nan)
    _end_time: float = pydantic.PrivateAttr(math.nan)
    _logger: logging.Logger = pydantic.PrivateAttr()
    _level: int = pydantic.PrivateAttr()

    def __init__(self, level=logging.DEBUG, **kwds):
        super().__init__(**kwds)
        assert level >= 0 and isinstance(level, int)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._level = level

    def set(self):
        self._start_time = time.perf_counter()

    def reset(self):
        self._start_time = math.nan
        self._end_time = math.nan

    def lap(self):
        assert not math.isnan(self._start_time), "Call set() first!"
        self._end_time = time.perf_counter()
        self._logger.log(self._level, f'{self.name} takes {self.duration:.2e} seconds.')

    @property
    def duration(self) -> float:
        return self._end_time - self._start_time


class ContextProfiler(Profiler):
    def __enter__(self):
        self.set()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lap()
    

class FunctionProfiler(Profiler):
    _fn: typing.Callable = pydantic.PrivateAttr()

    def __call__(self, fn: typing.Callable):
        assert callable(fn)
        self._fn = fn
        def wrapper(*args, **kwds):
            try:
                self.set()
                res = self._fn(*args, **kwds)
            finally:
                self.lap()
            return res
        return wrapper


class IteratorProfiler(Profiler):
    iterable: typing.Iterable
    _iterator: typing.Iterator = pydantic.PrivateAttr(None)
    _average: float = pydantic.PrivateAttr(0)
    _count: int = pydantic.PrivateAttr(0)

    def __iter__(self):
        self._iterator = self.iterable.__iter__()
        return self
    
    def __next__(self):
        try:
            self.set()
            element = next(self._iterator)
            self.lap()
            self._average += self.duration
            self._count += 1
        except StopIteration:
            if self._count != 0:
                self._logger.debug(f'{self.name} averages {self._average / self._count:.2e} seconds.')
            self.reset()
            raise StopIteration
        return element
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    @FunctionProfiler(name='simple function')
    def foo():
        return 2

    with ContextProfiler(name='simple context'):
        foo()

    for x in IteratorProfiler(name='iterator', iterable=range(5)):
        pass