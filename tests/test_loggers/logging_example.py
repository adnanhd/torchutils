# logging_example.py

import yaml
import logging
import logging.config

import abc


class TrainerLogger(metaclass=abc.ABCMeta):
    __slots__ = ['config', 'handler']

    def __init__(self, handler):
        self.handler = handler


class FileLogger(TrainerLogger):
    def __init__(
            self,
            filename='experiment.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ):
        super(FileLogger, self).__init__(logging.FileHandler(filename))
        self.handler.setLevel(level=level)
        self.formatter = logging.Formatter(format)
        self.handler.setFormatter(self.formatter)


class ConsoleLogger(TrainerLogger):
    def __init__(
            self,
            level=logging.DEBUG,
            format='%(name)s - %(levelname)s - %(message)s'
    ):
        super(ConsoleLogger, self).__init__(logging.StreamHandler())
        self.handler.setLevel(level=level)
        self.formatter = logging.Formatter(format)
        self.handler.setFormatter(self.formatter)


class ScoreLogger(object):
    pass


# Create a custom logger
logger = logging.getLogger(__name__)

# Add handlers to the logger
logger.addHandler(FileLogger().handler)
logger.addHandler(ConsoleLogger().handler)

if __name__ == '__main__':
    logger.warning('This is a warning')
    logger.error('This is an error')
