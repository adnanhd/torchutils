# Copyright © 2021 Chris Hughes
import abc
import logging
import pydantic
import typing
from .buffer import AverageScoreSender, AverageScoreReceiver


class TrainerBaseModel(pydantic.BaseModel):
    _logger: logging.Logger = pydantic.PrivateAttr()
    _sender: AverageScoreSender = pydantic.PrivateAttr()
    _recver: AverageScoreReceiver = pydantic.PrivateAttr()
    _buffer: typing.Dict[str, typing.Any] = pydantic.PrivateAttr(default_factory=dict)
    """
    The abstract base class to be subclassed when creating new callbacks.
    """
    def __init__(self,
                 level: int = logging.INFO,
                 writable_scores: typing.Set[str] = set(),
                 readable_scores: typing.Set[str] = set(),
                 **kwds):
        super().__init__(**kwds)
        assert isinstance(writable_scores, set) and isinstance(readable_scores, set)
        self._logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)
        self._sender = AverageScoreSender(names=writable_scores.copy())
        self._recver = AverageScoreReceiver(names=readable_scores.copy())

    def log(self, msg, level=logging.INFO, **kwds) -> None:
        self._logger.log(level=level, msg=msg, **kwds)

    def log_debug(self, msg, **kwds) -> None:
        self._logger.log(level=logging.DEBUG, msg=msg, **kwds)

    def log_info(self, msg, **kwds) -> None:
        self._logger.log(level=logging.INFO, msg=msg, **kwds)

    def log_warn(self, msg, **kwds) -> None:
        self._logger.log(level=logging.WARN, msg=msg, **kwds)

    def log_error(self, msg, **kwds) -> None:
        self._logger.log(level=logging.ERROR, msg=msg, **kwds)

    def log_fatal(self, msg, **kwds) -> None:
        self._logger.log(level=logging.FATAL, msg=msg, **kwds)

    @typing.final
    def add_handlers(self, handlers) -> None:
        logger: logging.Logger = self.__pydantic_private__['_logger']
        for hdlr in handlers:
            logger.addHandler(hdlr)

    @typing.final
    def remove_handlers(self, handlers) -> None:
        logger: logging.Logger = self.__pydantic_private__['_logger']
        for hdlr in handlers:
            logger.removeHandler(hdlr)

    @typing.final
    def add_score(self, name: str) -> None:
        sender: AverageScoreSender = self.__pydantic_private__['_sender']
        sender.add_score_names(name)

    @typing.final
    def add_score_names(self, name: str) -> None:
        recver: AverageScoreSender = self.__pydantic_private__['_recver']
        recver.add_score_names(name)

    @typing.final
    def log_score(self, name: str, value: float) -> None:
        self._sender.send(name, value)

    @typing.final
    def get_score_value(self) -> typing.Dict[str, float]:
        return self._recver.values

    @typing.final
    def get_score_averages(self) -> typing.Dict[str, float]:
        return self._recver.averages