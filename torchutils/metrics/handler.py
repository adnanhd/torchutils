from .tracker import AverageMeter, AverageMeterFunction
from collections import OrderedDict
from .history import RunHistory, DataFrameRunHistory
import warnings
import typing


class MetricHandler(object):
    from .tracker import MetricRegistrar
    __slots__ = ['_score_names', '_history', '_callbacks']
    DEFAULT_RUN_HISTORY = DataFrameRunHistory

    def __init__(self):
        self._score_names: typing.Set[str] = set()
        self._history: typing.Optional[RunHistory] = DataFrameRunHistory()
        self._callbacks: typing.List = list()

    @property
    def __scores__(self) -> typing.Dict[str, AverageMeter]:
        return {score_name: score_meter
                for score_name, score_meter in
                self.MetricRegistrar.__score__.items()
                if score_name in self._score_names}

    def get_score_names(self) -> typing.Set[str]:
        """ Returns score names of score meters (in the score list) """
        return set(self.__scores__.keys())

    def set_score_names(self, score_names: typing.Iterable[str]) -> None:
        """ Sets score names of score meters (in the score list) """
        self._history.set_score_names(score_names)
        self._score_names.clear()
        for score_name in score_names:
            if score_name is None:
                warnings.warn(f"{score_name} is not a registered score")
            elif score_name in self._score_names:
                warnings.warn(f"{score_name} is already in the score names")
            else:
                self._score_names.add(score_name)

    def reset_score_names(self) -> None:
        """ Clears score names of score meters (in the score list) """
        self._history.set_score_names([])
        self._score_names.clear()

    def add_score_meters(self, *scores: AverageMeter) -> None:
        """ Append a score meter to the score list """
        warnings.warn(
            "this method is no longer used, use "
            "MetricRegistrar.register_meter instead", DeprecationWarning
        )

    def remove_score_meters(self, *scores: str) -> None:
        """ Removes a score meter from the score list """
        warnings.warn(
            "this method is no longer used, use "
            "MetricRegistrar.unregister_meter instead", DeprecationWarning
        )

    def run_score_functional(self, preds, target):
        """ Runs callbacks hooked by add_score_group method """
        for cb in self._callbacks:
            cb.update(preds=preds, target=target)

    # AverageMeter functions
    def set_scores_values(self, **kwargs: float) -> None:
        """ Sets score values of the latest step """
        for score_name, score_value in kwargs.items():
            try:
                self.__scores__[score_name].update(value=score_value)
            except KeyError:
                warnings.warn(f"{score_name} is not in the registered"
                              "scores. Make sure that you passed the "
                              "key to set_score_names method beforehand.")

    def get_score_values(self, *score_names: str) -> typing.Dict[str, float]:
        """ Returns score values of the latest step """
        if len(score_names) == 0:
            score_names = self.get_score_names()
        # returns step result
        return {name: meter.value for name,
                meter in self.__scores__.items()}

    def reset_score_values(self) -> None:
        """ Resets score values of the step tracker """
        for score_meter in self.__scores__.values():
            score_meter.reset()

    # RunHistory functions
    def init_score_history(self, *score_names: str, fmt=':f') -> None:
        """ Sets score names of the epoch history """
        if len(score_names) == 0:
            score_names = self.get_score_names()
        self._history.set_score_names(score_names)

    def push_score_values(self, index: int = None) -> None:
        """ Stamps the score values of the latest epoch """
        for name in self._history.get_score_names():
            self._history.set_latest_score(name, self.__scores__[name].average)
        self._history._increment_epoch()

    def seek_score_history(self, *score_names: str) -> typing.Dict[str, float]:
        """ Returns the score values of the latest epoch """
        return {score_name: self._history.get_latest_score(score_name)
                for score_name in score_names}

    def get_score_history(
        self, *score_names: str
    ) -> typing.Dict[str, typing.Iterable[float]]:
        """ Returns the score values of all epochs """
        if len(score_names) == 0:
            score_names = self.get_score_names()
        return {score_name: self._history.get_score_values(score_name)
                for score_name in score_names}

    def reset_score_history(self) -> None:
        """ Clears all score values in the epoch history """
        self._history.reset_scores()
