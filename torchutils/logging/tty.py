from typing import overload
from .handler import TrainerLogger
import logging

class PrintWriter(TrainerLogger):
    """
    A callback which visualises the state of each training and evaluation epoch using a progress bar
    """
    filename = None
    log_format = '[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s'
    logging.basicConfig(format=log_format, filename=filename, level=logging.INFO)

    def __init__(self, total=None):
        super(PrintWriter, self).__init__()
        self._logger = None
        self._logs_dict = dict()

    def open(self, name='', **kwargs):
        self._logger = logging.getLogger(name=name, **kwargs)

    def log(self, **kwargs):
        self._logs_dict.update(kwargs)
        #os.sys.stdout.write(*args, ", ".join(f'{key}={value}' for key, value in kwargs.items()))

    def update(self, *args, **kwargs):
        self._logger.info(' '.join(f'{k}={v}' for k,v in self._logs_dict.items()))

    def close(self):
        del self._logger
        self._logger = None


class EpochPrintWriter(PrintWriter):
    def open(self, epoch):
        super().open(name=f'Epoch {epoch}')
