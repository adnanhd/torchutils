from torchutils.utils.pydantic import HandlerArguments, TrainerStatus
from collections import defaultdict
from typing import List, Dict
from abc import ABC
from .base import TrainerLogger
from .proxy import LoggingEvent, LoggerProxy


class LoggerHandler(ABC):
    __slots__ = ['_logger_dict_', '_logger_list_', '_current_event_']
    __handler__ = None

    def __init__(self):
        self._logger_dict_: Dict[LoggingEvent,
                                 List[TrainerLogger]] = defaultdict(list)
        self._logger_list_: List[TrainerLogger] = list()
        self._current_event_: List[LoggingEvent] = [None]

    def add_logger(self,
                   event: LoggingEvent,
                   logger: TrainerLogger):
        if not isinstance(event, LoggingEvent):
            raise AssertionError(
                f"{event} must be of LoggingEvent")
        elif isinstance(logger, TrainerLogger):
            self._logger_dict_[event].append(logger)
            self._logger_list_.append(logger)
        else:
            raise AssertionError(f"{logger} must be of TrainerLogger")

    def remove_logger(self,
                      event: LoggingEvent,
                      logger: TrainerLogger):
        if not isinstance(event, LoggingEvent):
            raise AssertionError(
                f"{event} must be of LoggingEvent")
        elif isinstance(logger, TrainerLogger):
            self._logger_dict_[event].remove(logger)
            self._logger_list_.remove(logger)
        else:
            raise AssertionError(f"{logger} must be of TrainerLogger")

    def set_event(self, event):
        assert isinstance(event, LoggingEvent)
        self._current_event_[0] = event

    def set_status(self, status: TrainerStatus) -> None:
        assert isinstance(status, TrainerStatus)
        self.setStatus(status)

    def clear_loggers(self):
        self._logger_dict_.clear()
        self._logger_list_.clear()

    def initialize(self, args: HandlerArguments, event: LoggingEvent = None):
        if event is None:
            loggers = self._logger_list_
        else:
            loggers = self._logger_dict_[event]

        for logger in loggers:
            logger.open(args)

    @classmethod
    def getHandler(cls):
        if cls.__handler__ is None:
            cls.__handler__ = cls()
        return cls.__handler__

    @classmethod
    def getProxy(cls) -> LoggerProxy:
        handler: LoggerHandler = cls.getHandler()
        return LoggerProxy(loggers=handler._logger_dict_,
                           _event_ptr=handler._current_event_)

    @classmethod
    def setStatus(cls, status: TrainerStatus) -> None:
        LoggerProxy.__status__[0] = status

    def terminate(self, stats: TrainerStatus, event: LoggingEvent = None):
        if event is None:
            loggers = self._logger_list_
        else:
            loggers = self._logger_dict_[event]

        for logger in loggers:
            logger.close(stats)
