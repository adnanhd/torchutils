import typing
import argparse
import logging
from .utils import DataFrame
from .utils import LoggingEvent
from torchutils.logging.base import TrainerLogger
from torchutils.trainer.utils import HandlerArguments, TrainerStatus


class SlurmLogger(TrainerLogger):
    __slots__ = ['_logger', '_level', '_handler', '_log_dict_']

    def __init__(self,
                 experiment: str,
                 host: typing.Optional[str] = None,
                 port: typing.Optional[float] = None,
                 format: str = '[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s',
                 filename: typing.Optional[str] = None,
                 level=logging.INFO, **config):
        self._level: int = level
        self._log_dict_ = dict()
        self._logger: logging.Logger = None
        if filename is not None:
            self._handler = logging.FileHandler(filename)
        elif host is not None:
            self._handler = logging.handlers.SocketHandler(host, port)
        else:
            self._handler = logging.NullHandler(level=level)
        logging.basicConfig(format=format, level=level, **config)

    @classmethod
    def getLogger(cls, event: LoggingEvent,
                  **kwargs) -> "TrainerLogger":
        if event == LoggingEvent.TRAINING_EPOCH:
            kwargs.setdefault('level', logging.INFO)
            return EpochSlurmLogger(**kwargs)
        else:
            kwargs.setdefault('level', logging.WARN)
            return BatchSlurmLogger(**kwargs)

    def open(self, args: HandlerArguments):
        self._logger = logging.getLogger()
        self._logger.addHandler(self._handler)

    def log_scores(self,
                   scores: typing.Dict[str, float],
                   status: TrainerStatus):
        self._log_dict_.update(scores)

    def update(self, n, status: TrainerStatus):
        pass

    def log_string(self,
                   string: str,
                   status: TrainerStatus):
        self._logger.log(self._level, string)

    def close(self, status: TrainerStatus):
        self._handler.close()


class EpochSlurmLogger(SlurmLogger):
    def open(self, args: HandlerArguments):
        name = f'Epoch {args.status.current_epoch}'
        self._logger = logging.getLogger(name=name)
        self._logger.addHandler(self._handler)

    def log_hyperparams(self,
                        params: argparse.Namespace,
                        status: TrainerStatus):
        self._log_dict_.update(params.__dict__)

    def log_table(self,
                  tables: typing.Dict[str, DataFrame],
                  status: TrainerStatus):
        self._log_dict_.update(tables)

    def update(self, n, status: TrainerStatus):
        self._logger.log(self._level, self._log_dict_)
        self._log_dict_.clear()


class BatchSlurmLogger(SlurmLogger):
    def log_scores(self,
                   scores: typing.Dict[str, float],
                   status: TrainerStatus):
        pass
