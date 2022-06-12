from torchutils.utils.pydantic import CurrentIterationStatus
from torchutils.trainer.loss import LossTracker
from torchutils.metrics import MetricHandler
import numpy as np

metrics = MetricHandler()
metrics._add_scores(['loss'])
loss_tracker: LossTracker = metrics.get_metric_instance('LossTracker')
proxy = CurrentIterationStatus(metrics)
proxy.set_score_names('loss')
# an acceptable error margin
epsilon = 1e-10


def random_iteration_arrays():
    x = np.zeros((100,))
    y_pred = np.random.rand(100)
    y_true = np.ones((100,))
    return x, y_true, y_pred


def random_step(num_epochs):
    assert proxy._score_history.get_score_values('loss') == []

    total_loss = 0
    for _ in range(num_epochs):
        x, y, y_pred = random_iteration_arrays()
        loss_tracker._loss = np.square(y_pred).sum().item()
        proxy.set_current_scores(x, y, y_pred)

        loss = metrics.get_score_values('loss')['loss']
        total_loss += loss
        assert proxy._score_history.get_score_values('loss') == []
        assert abs(proxy._score_tracker['loss'].score_value - loss) < epsilon

    proxy.average_scores()
    assert abs(proxy.get_latest_scores('loss')[
               'loss'] - total_loss / num_epochs) < epsilon
    proxy.reset_score_values()


def test_step_1():
    random_step(10)


def test_step_2():
    random_step(20)
