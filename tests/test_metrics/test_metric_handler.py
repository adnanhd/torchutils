from torchutils._dev_utils import MetricHandler, MeterBuffer
import pytest
from .metrics import *
import torch


@pytest.fixture
def get_input_output():
    torch.manual_seed(42)
    y_pred = torch.rand(100, 10)
    y_true = torch.randint(0, 10, (100,))
    return y_pred, y_true


def test_metric_handler(get_input_output):
    y_pred, y_true = get_input_output
    hdlr = MetricHandler(metrics={'accuracy_score'})
    hdlr.compute(batch=(None, y_true), batch_output=y_pred)
    res = MeterBuffer(scores={'accuracy_score'})
    assert res.obtain_score_average('accuracy_score') == 0.0900
    # assert isinstance(res['accuracy_score'], float)


def test_numpy_metric_handler(get_input_output):
    y_pred, y_true = get_input_output
    hdlr = MetricHandler(metrics={'numpy_accuracy_score'})
    hdlr.compute(batch=(None, y_true), batch_output=y_pred)
    res = MeterBuffer(scores={'numpy_accuracy_score'})
    assert res.obtain_score_average('numpy_accuracy_score') == 0.0900
    # assert isinstance(res['numpy_accuracy_score'], float)


def test_torch_metric_handler(get_input_output):
    y_pred, y_true = get_input_output
    hdlr = MetricHandler(metrics={'torch_accuracy_score'})
    hdlr.compute(batch=(None, y_true), batch_output=y_pred)
    res = MeterBuffer(scores={'torch_accuracy_score'})
    assert res.obtain_score_average('torch_accuracy_score') == 0.0900
    # assert isinstance(res['torch_accuracy_score'], float)
