from sklearn.metrics import accuracy_score as acc
from torchutils._dev_utils import register_torch_metric, register_numpy_metric, MetricType
import torch
import numpy as np


@register_numpy_metric
def numpy_accuracy_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return acc(y_pred=y_pred.argmax(1), y_true=y_true)


@register_torch_metric
def torch_accuracy_score(output: torch.Tensor, target: torch.Tensor) -> float:
    return float(sum(output.argmax(1) == target)) / len(output)


@MetricType.register
def accuracy_score(batch_output: torch.Tensor,
                   batch_target: torch.Tensor,
                   **batch_extra_kwds) -> float:
    return float(sum(batch_output.argmax(1) == batch_target)) / len(batch_target)