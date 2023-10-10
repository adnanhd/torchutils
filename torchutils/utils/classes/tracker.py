# Copyright © 2021 Chris Hughes
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterable, Dict, Set


class RunHistory(ABC):
    """
    The abstract base class which defines the API for a :class:`~pytorch_accelerated.trainer.Trainer`'s run history.
    """

    @abstractmethod
    def get_score_names(self, *score_names: str) -> Iterable:
        """
        Return a set containing of all unique score names which are being tracked.
        :return: an iterable of the unique score names
        """
        pass

    @abstractmethod
    def get_score_values(self, score_name) -> Iterable:
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


class InMemoryRunHistory(RunHistory):
    """
    An implementation of :class:`RunHistory` which stores all recorded values in memory.
    """

    def __init__(self, *score_names, current_epoch=1):
        self._current_epoch = current_epoch
        self._scores = defaultdict(list).fromkeys(score_names)

    def get_score_names(self) -> Set:
        return set(self._scores.keys())

    def get_score_values(self, score_name):
        return self._scores[score_name]

    def get_latest_score(self, score_name):
        if len(self._scores[score_name]) > 0:
            return self._scores[score_name][-1]
        else:
            raise ValueError(
                f"No values have been recorded for the score {score_name}"
            )

    def set_latest_score(self, score_name, score_value):
        self._scores[score_name].append(score_value)

    @property
    def current_epoch(self):
        return self._current_epoch

    def _increment_epoch(self, n=1):
        self._current_epoch += n

    def reset_scores(self, new_epoch=1):
        self._current_epoch = new_epoch
        self._scores = defaultdict(list)


class DataFrameRunHistory(RunHistory):
    """
    An implementation of :class:`RunHistory` which stores all recorded values in memory.
    """
    __slots__ = ['_current_epoch', '_scores', '_has_latest_row']

    def __init__(self, current_epoch: int = 1):
        self._current_epoch: int = current_epoch
        self._scores: pd.DataFrame = None
        self._has_latest_row = False

    def set_score_names(self, score_names):
        self._scores = pd.DataFrame(columns=score_names)
        self._has_latest_row = False

    def get_score_names(self) -> Set:
        return set(self._scores.columns)

    def get_score_values(self, score_name):
        return self._scores[score_name].to_list()

    def get_latest_score(self, score_name):
        if score_name in self._scores.columns:
            if self._scores[score_name].__len__() != 0:
                return self._scores[score_name].iloc[-1]
            else:
                raise IndexError(
                    f"No value has been set previously for the score {score_name}"
                )
        else:
            raise ValueError(
                f"The score {score_name} is not valid for RunHistory"
            )

    def set_latest_score(self, score_name, score_value):
        if not self._has_latest_row:
            self._append_epoch_row()
        self._scores[score_name].loc[self._current_epoch] = score_value

    @property
    def current_epoch(self):
        return self._current_epoch

    def _append_epoch_row(self):
        self._scores.loc[self._current_epoch] = np.nan
        self._has_latest_row = True

    def _increment_epoch(self, n=1):
        self._current_epoch += n
        self._has_latest_row = False

    def reset_scores(self, new_epoch=1):
        self._current_epoch = new_epoch
        self._scores.drop(self._scores.index, inplace=True)
        self._has_latest_row = False


class SingleScoreTracker:
    def __init__(self):
        self.score_value = 0
        self.total_score = 0
        self.running_count = 0

    def reset(self):
        self.score_value = 0
        self.total_score = 0
        self.running_count = 0

    def update(self, score_batch_value, batch_size=1):
        self.score_value = score_batch_value
        self.total_score += score_batch_value * batch_size
        self.running_count += batch_size

    @property
    def average(self):
        if self.running_count == 0:
            return np.nan
        return self.total_score / self.running_count


class ScoreTracker():
    def __init__(self):
        self._trackers = dict()
        self._runhists = DataFrameRunHistory()

    def set_score_names(self, score_names: Iterable):
        self._trackers = {score: SingleScoreTracker() for score in score_names}
        self._runhists.set_score_names(score_names)

    def get_score_names(self) -> Set:
        return set(self._trackers.keys())

    def set_current_scores(self, scores: Dict, batch_size=1):
        for score_name, score_value in scores.items():
            self._trackers[score_name].update(
                score_value, batch_size=batch_size)

    def set_averaged_scores(self, n=1):
        for score_name, score_tracker in self._trackers.items():
            self._runhists.set_latest_score(score_name, score_tracker.average)
            score_tracker.reset()
        self._runhists._increment_epoch(n)

    def get_averaged_scores(self, scores: Iterable, batch_size=1):
        if len(scores) == 0:
            scores = iter(self._trackers.keys())
        print(scores)
        return dict(map(self._runhists.get_latest_score, scores))

    def reset_scores(self, start_index=1):
        self._runhists.reset_scores(start_index)
