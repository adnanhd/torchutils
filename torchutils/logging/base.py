from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
import argparse


class LoggerMethodNotImplError(Exception):
    pass


class TrainerLogger(ABC):
    """
    A callback which visualises the state of each training
    and evaluation epoch using a progress bar
    """
    @abstractmethod
    def open(self):
        raise LoggerMethodNotImplError

    @abstractmethod
    def log_scores(self,
                   scores: Dict[str, float],
                   step: Optional[int] = None):
        raise LoggerMethodNotImplError

    def log_hyperparams(self,
                        params: argparse.Namespace,
                        *args,
                        **kwargs):
        raise LoggerMethodNotImplError

    def log_table(self,
                  key: str,
                  table: Dict[str, List[Any]]):
        raise LoggerMethodNotImplError

    def log_image(self,
                  key: str,
                  Images: List[Any],
                  step: Optional[int] = None):
        raise LoggerMethodNotImplError

    def log_string(self, msg):
        raise LoggerMethodNotImplError

    def log_module(self, module):
        raise LoggerMethodNotImplError

    @abstractmethod
    def close(self):
        raise LoggerMethodNotImplError
