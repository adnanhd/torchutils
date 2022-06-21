from .tracker import AverageMeter
from collections import OrderedDict
from .history import RunHistory, DataFrameRunHistory
import typing


class MetricHandler(object):
    __slots__ = ['_scores', '_history', '_callbacks']
    DEFAULT_RUN_HISTORY = DataFrameRunHistory

    def __init__(self):
        self._scores: typing.Dict[str, AverageMeter] = OrderedDict()
        self._history: typing.Optional[RunHistory] = DataFrameRunHistory()
        self._callbacks: typing.List = list()

    def get_score_names(self) -> typing.Set[str]:
        """ Returns score names of score meters (in the score list) """
        return set(self._scores.keys())

    def set_score_names(self, score_names: typing.Iterable[str]) -> None:
        """ Sets score names of score meters (in the score list) """
        self._history.set_score_names(score_names)

    def reset_score_names(self) -> None:
        """ Clears score names of score meters (in the score list) """
        self._history.set_score_names([])

    def add_score_meters(self, *scores: AverageMeter) -> None:
        """ Append a score meter to the score list """
        assert all(
            map(lambda o: isinstance(o, AverageMeter), scores)
        ), """values must be an instance of AverageMeter"""
        for score_meter in scores:
            self._scores[score_meter.name] = score_meter

    def remove_score_meters(self, *scores: str) -> None:
        """ Removes a score meter from the score list """
        for score_name in scores:
            self._scores.pop(score_name)

    def add_score_group(self, *score_names: str):
        """ Appends a score function to callback list to call automatically """
        handler = self
        self._scores.update({
            name: AverageMeter(name)
            for name in score_names
        })

        class ScoreHook(object):
            __slots__ = ['score_names', 'fn', 'handler']

            def __init__(self, fn, score_names=score_names, handler=handler):
                self.fn = fn
                self.handler: MetricHandler = handler
                self.handler._callbacks.append(self)
                self.score_names: typing.List[str] = score_names

            def __call__(self, x, y, y_pred):
                scores = self.fn(x=x, y=y, y_pred=y_pred)
                self.handler.set_scores_values(**scores)

        return ScoreHook

    def run_score_groups(self, x, y, y_pred):
        """ Runs callbacks hooked by add_score_group method """
        for fn in self._callbacks:
            fn(x=x, y=y, y_pred=y_pred)

    # AverageMeter functions
    def set_scores_values(self, **kwargs: float) -> None:
        """ Sets score values of the latest step """
        for score_name, score_value in kwargs.items():
            self._scores[score_name].update(value=score_value)

    def get_score_values(self, *score_names: str) -> typing.Dict[str, float]:
        """ Returns score values of the latest step """
        if len(score_names) == 0:
            score_names = self.get_score_names()
        # returns step result
        return {name: meter.value for name,
                meter in self._scores.items()}

    def reset_score_values(self) -> None:
        """ Resets score values of the step tracker """
        for score_meter in self._scores.values():
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
            self._history.set_latest_score(name, self._scores[name].average)
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
