from .buffer import AverageScoreBuffer
from .metric import TrainerMetric
import typing


class AverageScoreHandler(AverageScoreBuffer):
    def reset_scores(self, *score_names: str):
        if len(score_names) == 0:
            score_names = self.names
        for score in map(self._get_average_score, self._names):
            score.reset()
            

class MetricHandler:
    """
    The :class:`MetricHandler` is responsible for calling a list of metrics.
    This class calls the metrics in the order that they are given.
    """
    __slots__ = ['metrics']

    def __init__(self, metrics=list()):
        self.metrics: typing.List[TrainerMetric] = metrics