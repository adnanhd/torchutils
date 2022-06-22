from torchutils.utils.pydantic import HandlerArguments
from torchutils.logging.base import TrainerLogger
from typing import Optional, Dict, List, Any
import logging


class PrintWriter(TrainerLogger):
    __slots__ = ['_logger', '_summary']

    def __init__(self,
                 filename: Optional[str] = None,
                 format: str = '[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s',
                 level=logging.INFO,
                 **kwargs):
        self._logger: logging.Logger = None
        logging.basicConfig(format=format,
                            filename=filename,
                            level=level,
                            **kwargs)
        self._summary = None

    def open(self, args: HandlerArguments = None, **kwargs):
        self._logger = logging.getLogger(**kwargs)

    def log_scores(self,
                   scores: Dict[str, float],
                   step: Optional[int] = None):
        self._logger.info(scores)

    def log_table(self,
                  key: str,
                  table: Dict[str, List[Any]]):
        self._logger.info(str(table))

    def log_string(self, msg):
        self._logger.info(msg)

    # TODO: add summary to log_module
    # def log_module(self, module):
    #     raise LoggerMethodNotImpl()

    def close(self):
        self._logger = None
