from typing import overload, List
from abc import ABC, abstractmethod


class TrainerLogger(ABC):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """
    @abstractmethod
    def open(self):
        ...

    @abstractmethod
    def log_score(self, **kwargs):
        ...

    @abstractmethod
    def log_info(self, msg):
        ...
    
    @abstractmethod
    def log_error(self, msg):
        ...

    @abstractmethod
    def _flush_step(self):
        ...

    @abstractmethod
    def _flush_epoch(self):
        ...

    @abstractmethod
    def close(self):
        ...


class LoggingHandler(ABC):
    def __init__(self, loggers=[]):
        self._loggers: List[TrainerLogger] = []
        self.add_loggers(loggers)

    def add_logger(self, logger):
        if isinstance(logger, trainerlogger):
            self._loggers.append(logger)

    def remove_logger(self, logger):
        self._loggers.remove(logger)

    def add_loggers(self, loggers: List[TrainerLogger]):
        for logger in loggers:
            if isinstance(logger, trainerlogger):
                self._loggers.extend(loggers)

    def clear_loggers(self):
        self._loggers.clear()

    def init(self):
        for logger in self._loggers:
            logger.open()

    def step(self):
        for logger in self._loggers:
            logger._flush_step()

    def epoch(self):
        for logger in self._loggers:
            logger._flush_epoch()

    def end(self):
        for logger in self._loggers:
            logger.close()

