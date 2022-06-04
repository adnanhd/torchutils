from typing import overload, List, Dict
from abc import ABC, abstractmethod
from torchutils.utils.pydantic import HandlerArguments, EpochResults, StepResults, TrainerModel


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

    def log_info(self, msg):
        ...
    
    def log_error(self, msg):
        ...

    def log_model(self, model):
        ...

    @abstractmethod
    def close(self):
        ...


class LoggingHandler(ABC):
    def __init__(self, loggers=[]):
        self._loggers: List[TrainerLogger] = []
        self.add_loggers(loggers)

    def add_logger(self, logger):
        if isinstance(logger, TrainerLogger):
            self._loggers.append(logger)

    def remove_logger(self, logger):
        self._loggers.remove(logger)

    def add_loggers(self, loggers: List[TrainerLogger]):
        for logger in loggers:
            self._loggers.add_logger(loggers)

    def clear_loggers(self):
        self._loggers.clear()

    def initialize(self, args: HandlerArguments):
        for logger in self._loggers:
            logger.open(args)

    def model(self, model: TrainerModel):
        for logger in self._loggers:
            logger.log_model(model)

    def score(self, **scores):
        for logger in self._loggers:
            logger.log_score(**scores)

    def figure(self, msg):
        pass

    def terminate(self):
        for logger in self._loggers:
            logger.close()
