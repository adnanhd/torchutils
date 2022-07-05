# Copyright © 2021 Chris Hughes
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import typing


class RunHistory(ABC):
    """
    The abstract base class which defines the API for a :class:`~pytorch_accelerated.trainer.Trainer`'s run history.
    """

    @abstractmethod
    def set_score_names(self, score_names: typing.Iterable[str]) -> None:
        """ Sets score names of the history """
        pass

    @abstractmethod
    def get_score_names(self) -> typing.Set[str]:
        """
        Return a set containing of all unique score names which are being tracked.
        :return: an iterable of the unique score names
        """
        pass

    @abstractmethod
    def get_score_values(self, score_name) -> typing.Iterable:
        """
        Return all of the values that have been recorded for the given score.
        :param score_name: the name of the score being tracked
        :return: an ordered iterable of values that have been recorded for that score
        """
        pass

    @abstractmethod
    def get_latest_score(self, score_name):
        """
        Return the most recent value that has been recorded for the given score.
        :param score_name: the name of the score being tracked
        :return: the last recorded value
        """
        pass

    @abstractmethod
    def set_latest_score(self, score_name, score_value):
        """
        Record the value for the given score.
        :param score_name: the name of the score being tracked
        :param score_value: the value to record
        """
        pass

    @property
    @abstractmethod
    def current_epoch(self) -> int:
        """
        Return the value of the current epoch.
        :return: an int representing the value of the current epoch
        """
        pass

    @abstractmethod
    def _increment_epoch(self):
        """
        Increment the value of the current epoch
        """
        pass

    @abstractmethod
    def reset_scores(self):
        """
        Reset the state of the :class:`RunHistory`
        """
        pass


class DataFrameRunHistory(RunHistory):
    """
    An implementation of :class:`RunHistory` which stores all recorded values in memory.
    """
    __slots__ = ['_current_epoch', '_scores', '_has_latest_row']

    def __init__(self,
                 score_names: typing.Set[str] = set(),
                 current_epoch: int = 1):
        self._current_epoch: int = current_epoch
        self._scores: pd.DataFrame = None
        self._has_latest_row = False
        self.set_score_names(score_names)

    @property
    def current_epoch(self):
        return self._current_epoch

    def set_score_names(self, score_names: typing.Iterable[str]) -> None:
        """ Sets score names of the history """
        self._scores = pd.DataFrame(columns=score_names)
        self._has_latest_row = False

    def get_score_names(self) -> typing.Set[str]:
        """ Returns score names """
        return set(self._scores.columns)

    def get_score_values(self, score_name: str) -> typing.List[float]:
        """ Returns score values as a list """
        return self._scores[score_name].to_list()

    def get_latest_score(self, score_name) -> float:
        """ Returns last entry of which score name is given """
        if score_name not in self._scores.columns:
            raise ValueError(
                f"The score {score_name} is not valid for RunHistory"
            )
        elif self._scores[score_name].__len__() == 0:
            raise IndexError(
                f"No value has been set previously for the score {score_name}"
            )
        else:
            return self._scores[score_name].iloc[-1]

    def set_latest_score(self, score_name: str, score_value: float) -> None:
        """ Sets last entry of which score name is given """
        if not self._has_latest_row:
            self._allocate_epoch()
            # warnings.warn(
            #     f"Please call {self.__class__.__name__}._append_row "
            #     " to initialize current row", DeprecationWarning
            # )
        self._scores[score_name].loc[self._current_epoch] = score_value

    def _allocate_epoch(self):
        self._scores.loc[self._current_epoch] = np.nan
        self._has_latest_row = True

    def _increment_epoch(self, n: int = 1):
        """ Appends an entry to each scores in the history """
        self._current_epoch += n
        self._has_latest_row = False

    def reset_scores(self, new_epoch: int = 1):
        """ Clears all entries in the history """
        self._current_epoch = new_epoch
        self._scores.drop(self._scores.index, inplace=True)
        self._has_latest_row = False
