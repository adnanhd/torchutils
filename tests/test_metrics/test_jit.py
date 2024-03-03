from torchutils._dev_utils import MetricHandler
from torchutils._dev_utils import MetricType
import torch
import pytest
import typing
import ray


@pytest.fixture
def get_input_output():
    torch.manual_seed(42)
    y_pred = torch.rand(100,10)
    y_true = torch.randint(0, 10, (100,))
    return y_pred, y_true

@MetricType.register
#@torch.jit.script
def jit_accuracy_score(batch_output: torch.Tensor,
                   batch_target: torch.Tensor) -> float:
    return float(sum(batch_output.argmax(1) == batch_target)) / len(batch_target)

@MetricType.register
@ray.remote
def ray_accuracy_score(batch_output: torch.Tensor,
                       batch_target: torch.Tensor) -> float:
    return float(batch_output.argmax(0) == batch_target)


def test_metric_handler(get_input_output):
    y_pred, y_true = get_input_output
    hdlr = MetricHandler(metrics={'ray_accuracy_score'})
    res = hdlr.compute(batch=(None, y_true), batch_output=y_pred)
    assert res['ray_accuracy_score'] == 0.0900
    assert isinstance(res['ray_accuracy_score'], float)