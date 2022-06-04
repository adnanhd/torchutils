from torchutils.utils.pydantic import HandlerArguments
from torchutils.logging import TrainerLogger
from collections import OrderedDict
from typing import Optional
import logging

class PrintWriter(TrainerLogger):
    __slots__ = ['_logger']
    def __init__(self, 
            filename: Optional[str] = None,
            format: str = '[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s',
            level = logging.INFO, 
            **kwargs):
        self._logger: logging.Logger = None
        logging.basicConfig(format=format, 
                filename=filename, level=level, **kwargs)

    def open(self, **kwargs):
        #TODO: add config here
        self._logger = logging.getLogger(**kwargs)

    def open(self, args: HandlerArguments = None, **kwargs):
        self._logger = logging.getLogger(**kwargs)

    def log_score(self, **kwargs):
        self._logger.info(kwargs)

    def close(self):
        self._logger = None

