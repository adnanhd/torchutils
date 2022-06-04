from torchutils.logging import TrainerLogger
from collections import OrderedDict
import logging

class PrintWriter(TrainerLogger):
    filename = None
    log_format = '[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s'
    logging.basicConfig(format=log_format, filename=filename, level=logging.INFO)
    __slots__ = ['_logger']
    
    def open(self):
        #TODO: add config here
        self._logger = logging.getLogger(name=name, **kwargs)

    def log_score(self, **kwargs):
        self._logger.info(kwargs)

    def log_model(self, model):
        self._logger.info(model.summary())

    def log_error(self, msg: str):
        self._logger.error(msg)

    def log_info(self, msg: str):
        self._logger.info(msg)
    
    def _flush_step(self):
        pass

    def _flush_epoch(self):
        pass

    def close(self):
        self._logger = None
