# logging_example.py

import logging
import logging.config
from .base import ExperimentLogger


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
