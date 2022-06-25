from torchutils.utils.pydantic import CurrentIterationStatus
from torchutils.metrics import MetricHandler
from .utils import epsilon_equal
import numpy as np

handler = MetricHandler()
proxy = CurrentIterationStatus(handler)
handler.init_score_history('loss')
total_loss = []


@handler.add_score_group('loss')
def loss_score_group(x, y, y_pred):
    return {'loss': np.square(y_pred).sum().item()}


def random_iteration_arrays():
    x = np.zeros((100,))
    y_pred = np.random.rand(100)
    y_true = np.ones((100,))
    return x, y_true, y_pred


def random_step(num_batches):
    assert handler.get_score_history('loss')['loss'] == total_loss

    epoch_loss = 0
    for _ in range(num_batches):
        x, y, y_pred = random_iteration_arrays()
        proxy.set_current_scores(x, y, y_pred)

        loss = handler.get_score_values('loss')['loss']
        epoch_loss += loss
        assert epsilon_equal(handler.get_score_values('loss')['loss'], loss)
        assert epsilon_equal(proxy.get_current_scores('loss')['loss'], loss)

    # applies both push and reset scores
    proxy.average_current_scores()
    epoch_loss /= num_batches
    assert epsilon_equal(handler.seek_score_history('loss')[
                         'loss'], epoch_loss)
    total_loss.append(epoch_loss)
    assert epsilon_equal(proxy.get_score_history('loss')['loss'], total_loss)


def test_step_1():
    random_step(10)


def test_step_2():
    random_step(20)
