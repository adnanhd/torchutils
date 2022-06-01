from .base import TrainerMetrics
from typing import Set

class BinaryClassificationMetrics(TrainerMetrics):
    __slots__ = ['f1', 'accuracy', 'precision', 'recall',
                    '_tp', '_tn', '_fp', '_fn']

    #def set_metrics(self, x, y, y_pred):

    @property
    def metric_names(self) -> Set:
        return {'f1', 'accurcay', 'precision', 'recall'}
