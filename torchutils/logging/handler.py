from torchutils.trainer.utils import IterationArguments, IterationStatus
from collections import defaultdict
from typing import List, Dict
from abc import ABC
from .base import TrainerLogger
from .proxy import LoggingEvent, LoggerProxy


class LoggerHandler(ABC):
    __slots__ = ['_logger_dict_', '_logger_list_',
                 '_current_event_', '_current_status_']
    __handler__ = None

    def __init__(self):
        if self.__class__.__handler__ is None:
            self.__class__.__handler__ = self
            self._logger_dict_: Dict[LoggingEvent,
                                     List[TrainerLogger]] = defaultdict(list)
            self._logger_list_: List[TrainerLogger] = list()
            self._current_event_: List[LoggingEvent] = [None]
            self._current_status_: List[IterationStatus] = [None]
        else:
            handler = self.__class__.__handler__
            for attr_name in self.__class__.__slots__:
                attr = getattr(handler, attr_name)
                setattr(self, attr_name, attr)

    @classmethod
    def getHandler(cls):
        if cls.__handler__ is None:
            cls.__handler__ = cls()
        return cls.__handler__

    @classmethod
    def getProxy(cls) -> LoggerProxy:
        handler: LoggerHandler = cls.getHandler()
        return LoggerProxy(loggers=handler._logger_dict_,
                           _event_ptr=handler._current_event_,
                           _status_ptr=handler._current_status_)

    def add_loggers(self, loggers: Dict[TrainerLogger, LoggingEvent]):
        for logger, events in loggers.items():
            for event in events:
                self.add_logger(event=event, logger=logger)

    def add_logger(self,
                   event: LoggingEvent,
                   logger: TrainerLogger):
        assert isinstance(event, LoggingEvent), \
            f"{event} must be of LoggingEvent"
        assert isinstance(logger, TrainerLogger), \
            f"{logger} must be of TrainerLogger"
        self._logger_dict_[event].append(logger)
        self._logger_list_.append(logger)

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

    def set_status(self, status: IterationStatus) -> None:
        assert isinstance(status, IterationStatus)
        self._current_status_[0] = status

    def clear_loggers(self):
        self._logger_dict_.clear()
        self._logger_list_.clear()

    def initialize(self, args: IterationArguments, event: LoggingEvent = None):
        if event is None:
            loggers = self._logger_list_
        else:
            loggers = self._logger_dict_[event]

        if event == LoggingEvent.TRAINING_BATCH:
            for logger in loggers:
                logger.open(args, self._current_status_[0])
        else:
            for logger in loggers:
                logger.open(args)

    def terminate(self, stats: IterationStatus, event: LoggingEvent = None):
        if event is None:
            loggers = self._logger_list_
        else:
            loggers = self._logger_dict_[event]

        for logger in loggers:
            logger.close(stats)
