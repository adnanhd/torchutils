# PyTorch-Utils Metric Examples

## An Example with TrainerMetric inheritance
```python
from torchutils.metrics import TrainerMetric, register_to_metric_handler

# Equally 
# from torchutils.metrics import MetricHandler
# MetricHandler.register_metric(MyMetric())
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

## An Example with MetricHandler use
```py
from torchutils.metrics import MetricHandler

dataloader = load_my_dataloader()
model = build_my_model()
mh = MetricHandler('x_score', 'y_score')

for epoch_idx in range(10):
  for x, y in dataloader:
      y_pred = model(x)
      mh.set_scores_values(x, y, y_pred)
      print(mh.get_score_values())
```

after creating a custom TraierMetric, add one line to use it with torchutils.trainer API
```diff

trainer = Trainer(model=model, loss=loss, ...)
+ trainer.compile(metrics=['x_score', 'y_score'])
trainer.train(...)
```




