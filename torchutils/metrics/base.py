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
    _scores: Dict[Str, TrainerMetric] = OrderedDict()
    _metrics: Dict[Str, TrainerMetric] = OrderedDict()

    def __init__(self, *score_names):
        self._scores: Dict[Str, TrainerMetric] = OrderedDict()
        self._metrics: Dict[Str, TrainerMetric] = OrderedDict()
        # TODO: why it is here? all_score_names = self.__class__._scores.copy()
        self.add_scores(score_names)
    
    def add_scores(self, *score_names: Str):
        for score_name in score_names:
            if score_name not in self.__class__._scores.keys():
                warnings.warn(f"Score {score_name} is not registered", 
                        RuntimeWarning)
                continue
            trainer_metric = self.__class__._scores[score_name]
            self._scores[score_name] = trainer_metric 
            self._metrics[trainer_metric.__class__.__name__] = trainer_metric

    def remove_scores(self, *score_names: Str):
        for score_name in score_names:
            if score_name not in self.__class__._scores.keys():
                warnings.warn(f"Score {score_name} is not registered", 
                        RuntimeWarning)
                continue
            metric = self._scores.pop(score_name)
            if metric not in self._scores.values():
                self._metrics.pop(metric.__class__.__name__)

    def clear_scores(self):
        self._metrics.clear()
        self._scores.clear()

    #TODO: write register_scores
    @classmethod
    def register_metric(cls, trainer_metric):
        for score_name in trainer_metric.score_names:
            cls._scores[score_name] = trainer_metric 
            cls._metrics[trainer_metric.__class__.__name__] = trainer_metric

    #TODO: create class methods
    def get_score_names(self) -> Iterable:
        return iter(self._scores.keys())

    def get_metric_names(self) -> Iterable:
        return iter(self._metrics.keys())

    def get_metric_instance(self, metric_name):
        return self._metrics[metric_name]

    def get_score_values(self, *score_names) -> Mapping[Str, Any]:
        if len(score_names) == 0: score_names = self.get_score_names()
        return { score_name: self._scores[score_name] \
            .get_scores(score_name) for score_name in score_names }

    def set_scores_values(self, x, y, y_pred):
        for metric_name in self.get_metric_names():
            self._metrics[metric_name].set_scores(x, y, y_pred)

