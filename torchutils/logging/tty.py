# logging_example.py

import logging
import logging.config
from .base import ExperimentLogger


class FileLogger(ExperimentLogger):
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


class ConsoleLogger(ExperimentLogger):
    def __init__(
            self,
            level=logging.DEBUG,
            format='%(name)s - %(levelname)s - %(message)s'
    ):
        super(ConsoleLogger, self).__init__(logging.StreamHandler())
        self.handler.setLevel(level=level)
        self.formatter = logging.Formatter(format)
        self.handler.setFormatter(self.formatter)


if __name__ == '__main__':
    # Create a custom logger
    # @TODO: @Client i.e. Callback
    logger = logging.getLogger(__name__)

    # Add handlers to the logger
    # @TODO: @Server i.e. IterationInterface
    logger.addHandler(FileLogger('/tmp/logconfig.log').handler)
    logger.addHandler(ConsoleLogger().handler)

    logger.warning('This is a warning')
    logger.error('This is an error')
