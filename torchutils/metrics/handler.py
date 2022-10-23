import warnings
import typing
from .tracker import (
    AverageScore,
    AverageScoreModule,
    AverageScoreFunctional,
    FunctionalOrModule
)


class MetricHandler(object):
    from .tracker import MetricRegistrar
    __slots__ = ['_score_names']

    def __init__(self, *score_names: str):
        self._score_names: typing.Set[str] = set()
        self.set_score_names(score_names)

    @property
    def __scores__(self) -> typing.Dict[str, AverageScore]:
        return {
            score_name: self.MetricRegistrar.__score__[score_name]
            for score_name in self._score_names
        }
        # return self.MetricRegistrar.__score__

    @property
    def __callbacks__(self) -> typing.Set[FunctionalOrModule]:
        return {
            self.MetricRegistrar.__metrics__[
                self.MetricRegistrar.__score_to_metric__[score_name]
            ] for score_name in self._score_names
            if score_name in self.MetricRegistrar.__score_to_metric__
        }

    def add_score_meters(self, meters: typing.Iterable[AverageScore]):
        for meter in meters:
            assert isinstance(meter, AverageScore)
            if isinstance(meter, AverageScoreModule) \
                    or isinstance(meter, AverageScoreFunctional):
                self.MetricRegistrar.register_functional(meter)
            elif meter.name not in self.__scores__:
                self.MetricRegistrar.register_score(meter)
            else:
                raise KeyError(f"{meter.name} is already registered.")

    def get_score_names(self) -> typing.Set[str]:
        """ Returns score names of score meters (in the score list) """
        return set(self.__scores__.keys())

    def set_score_names(self, score_names: typing.Iterable[str]) -> None:
        """ Sets score names of score meters (in the score list) """
        # self._history.set_score_names(score_names)
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
        # self._history.set_score_names([])
        self._score_names.clear()

    def run_score_functional(self, preds, target):
        """ Runs callbacks hooked by add_score_group method """
        for cb in self.__callbacks__:
            cb.update(preds=preds, target=target)

    # AverageScore functions
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
        # returns step result
        return {name: meter.value for name,
                meter in self.__scores__.items()}

    def get_score_averages(self, *score_names: str) -> typing.Dict[str, float]:
        """ Returns average of score values by the time reset_score_values called """
        scores = self.__scores__
        # returns step result
        return {name: scores[name].average
                for name in score_names}

    def reset_score_values(self) -> None:
        """ Resets score values of the step tracker """
        for score_meter in self.__scores__.values():
            score_meter.reset()
