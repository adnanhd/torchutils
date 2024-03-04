from sklearn.metrics import accuracy_score as acc
from torchutils._dev_utils import register_numpy_metric
import numpy as np


@register_numpy_metric
def accuracy_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return acc(y_pred=y_pred.argmax(1), y_true=y_true)


np.random.seed(42)
x = np.random.rand(10, 5)
y = np.random.randint(0, 5, (10,))
