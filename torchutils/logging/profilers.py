import abc
import warnings
import typing
import logging
import logging.config


class TrainerProfiler(abc.ABC):
    __slots__ = ['config', 'handler']

    def __init__(self, handler):
        self.handler = handler


class FileProfiler(TrainerProfiler):
    def __init__(
            self,
            filename='experiment.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ):
        super(FileProfiler, self).__init__(logging.FileHandler(filename))
        self.handler.setLevel(level=level)
        self.formatter = logging.Formatter(format)
        self.handler.setFormatter(self.formatter)


class ConsoleProfiler(TrainerProfiler):
    def __init__(
            self,
            level=logging.WARNING,
            format='%(name)s - %(levelname)s - %(message)s'
    ):
        super(ConsoleProfiler, self).__init__(logging.StreamHandler())
        self.handler.setLevel(level=level)
        self.formatter = logging.Formatter(format)
        self.handler.setFormatter(self.formatter)


class ExperimentProfiler(abc.ABC):
    __slots__ = ['__loggers__']

    def __init__(self):
        self.__loggers__: typing.List[TrainerProfiler] = list()

    @property
    def handlers(self) -> typing.List[logging.Handler]:
        return [logger.handler for logger in self.__loggers__]

    def add_loggers(self, loggers: typing.List[TrainerProfiler]):
        for logger in loggers:
            self.add_logger(logger)

    def add_logger(self, logger: TrainerProfiler):
        assert isinstance(logger, TrainerProfiler)
        self.__loggers__.append(logger)

    def remove_loggers(self, loggers: typing.List[TrainerProfiler]):
        for logger in loggers:
            self.remove_logger(logger)

    def remove_logger(self, logger: TrainerProfiler):
        assert isinstance(logger, TrainerProfiler)
        if logger in self.__loggers__:
            self.__loggers__.remove(logger)
        else:
            warnings.warn(f"{logger} is not in loggers", UserWarning)

    def clear_loggers(self):
        self.__loggers__.clear()
