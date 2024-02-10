from sklearn.metrics import accuracy_score as acc
from torchutils._dev_utils import MetricHandler, MetricType, NumpyMetricType, MeterHandler
from torchutils._dev_utils.wrappers import log_score_on_return
import torch
import typing
import numpy as np



@NumpyMetricType.register
def numpy_sample_metric(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return 1

@MetricType.register
def accuracy_score(batch_output: torch.Tensor, batch_target: torch.Tensor, **batch_extra_kwds) -> float:
    return acc(y_pred=batch_output.argmax(1), y_true=batch_target)


hdlr = MetricHandler(metrics={accuracy_score})
y_pred = torch.rand(100,10)
y_true = torch.randint(0, 10, (100,))


hdlr.compute(batch=(None, y_true), batch_output=y_pred)
mt_hdlr = MeterHandler()