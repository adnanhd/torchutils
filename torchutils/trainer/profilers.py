import time
import typing
from ..metrics import TrainerAverageScore
    

class Profiler(object):
    def __init__(self, score: TrainerAverageScore):
        self.start_time: float = None
        self.end_time: float = None
        self.score: TrainerAverageScore = score

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()
        self.score.update(self.end_time - self.start_time)
    
    @property
    def duration(self):
        return self.end_time - self.start_time
    

class ProfilerFunctor(Profiler):
    def __init__(self, callable):
        self.callable: typing.Callable = callable
    
    def __call__(self, *args, **kwds):
        self.start()
        element = self.callable(*args, **kwds)
        self.stop()
        return element


class ProfilerIterator(Profiler):
    def __init__(self, iterable, score):
        super().__init__(score=score)
        self.iterable: typing.Iterable = iterable
        self.iterator: typing.Iterator = None

    def __iter__(self):
        self.iterator = self.iterable.__iter__()
        return self
    
    def __next__(self):
        self.start()
        element = next(self.iterator)
        self.stop()
        return element