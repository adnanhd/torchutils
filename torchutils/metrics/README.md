# Example use
```python
from typing import overload, Iterable, Callable
from torchutils.metrics import TrainerMetric, register_to_metric_handler

# Equally 
# from torchutils.metrics import MetricHandler
# @MetricHandler.register_metric
@register_to_metric_handler
class MyMetric(TrainerMetric):
    def set_scores(self, x, y, y_pred) -> None:
        self._my_score = ((y_pred - y).square()).mean(0).sum()
	self._my_score2 = x * y

    @property
    def score_names(self):
        return {'x_score', 'y_score'}

    @property
    def x_score(self):
        return self._my_score

    @property
    def y_score(self):
        return self._my_score2
```
	
