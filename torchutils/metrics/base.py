from abc import ABC, abstractmethod, abstractproperty, abstractclassmethod
from collections import OrderedDict
from typing import (
    Iterable,
    Set,
    Any,
    Dict,
    List,
    Tuple,
    Mapping,
    Union,
    Callable,
    NewType
)
import warnings
Str = NewType('Str', str)


class TrainerMetric(ABC):
    @abstractmethod
    def set_scores(self, x, y, y_pred):
        ...

    @abstractproperty
    def score_names(self) -> Union[Set, List, Tuple]:
        ...

    def get_scores(self, *score_names) -> Union[Mapping, Dict]:
        if len(score_names) == 0:
            score_names = self.score_names
        return { metric_name: getattr(self, metric_name) 
                for metric_name in score_names }

    def __call__(self, x, y, y_pred) -> Union[Mapping, Dict]:
        self.set_scores(x, y, y_pred)
        return self.get_scores(self.score_names)


class MetricHandler(object):
    _scores_by_name: Mapping[Str, TrainerMetric] = OrderedDict()
    _metric_by_name: Mapping[Str, TrainerMetric] = OrderedDict()

    def __init__(self, *score_names):
        self._scores_by_name: Mapping[Str, TrainerMetric] = OrderedDict()
        self._metric_by_name: Mapping[Str, TrainerMetric] = OrderedDict()
        all_score_names = self.__class__._scores_by_name
        for score_name in score_names:
            if score_name not in self.__class__._scores_by_name.keys():
                warnings.warn(f"Score {score_name} is not registered", RuntimeWarning)
            else:
                trainer_metric = self.__class__._scores_by_name[score_name]
                self._scores_by_name[score_name] = trainer_metric 
                self._metric_by_name[trainer_metric.__class__.__name__] = trainer_metric
    
    @classmethod
    def register_metric(cls, trainer_metric):
        for score_name in trainer_metric.score_names:
            cls._scores_by_name[score_name] = trainer_metric 
            cls._metric_by_name[trainer_metric.__class__.__name__] = trainer_metric

    #TODO: create class methods
    def get_score_names(self) -> Iterable:
        return iter(self._scores_by_name.keys())

    def get_metric_names(self) -> Iterable:
        return iter(self._metric_by_name.keys())

    def get_metric_instance(self, metric_name):
        return self._metric_by_name[metric_name]

    def get_score_values(self, *score_names) -> Mapping[Str, Any]:
        if len(score_names) == 0: score_names = self.get_score_names()
        return { score_name: self._scores_by_name[score_name] \
            .get_scores(score_name) for score_name in score_names }

    def set_scores_values(self, x, y, y_pred):
        for metric_name in self.get_metric_names():
            self._metric_by_name[metric_name].set_scores(x, y, y_pred)

