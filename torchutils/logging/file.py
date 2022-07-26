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
