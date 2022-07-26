import abc
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
