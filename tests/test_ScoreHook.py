from torchutils.metrics import MetricHandler
from .utils import epsilon_equal
import warnings
import numpy as np


handler = MetricHandler()
# add_score_group and init_score_history call
# order does not matter
handler.init_score_history('loss')


@handler.add_score_group("loss")
def score_group(x, y, y_pred):
    return {"loss": x}


total_loss = []


def test_ScoreGroupHook_epoch_1():
    assert not handler._history._has_latest_row
    assert handler._history.current_epoch == 1

    handler.run_score_groups(1e-2, None, None)
    assert epsilon_equal(handler.get_score_values('loss')['loss'], 1e-2)
    handler.run_score_groups(1e-3, None, None)
    assert epsilon_equal(handler.get_score_values('loss')['loss'], 1e-3)
    handler.run_score_groups(1e-4, None, None)
    assert epsilon_equal(handler.get_score_values('loss')['loss'], 1e-4)

    warnings.simplefilter("ignore", DeprecationWarning)
    handler.push_score_values()
    handler.reset_score_values()

    total_loss.append((1e-2 + 1e-3 + 1e-4) / 3)
    assert np.isnan(handler.get_score_values('loss')['loss'])
    assert epsilon_equal(handler.get_score_history('loss')['loss'], total_loss)


def test_ScoreGroupHook_epoch_2():
    handler.run_score_groups(1, None, None)
    assert epsilon_equal(handler.get_score_values('loss')['loss'], 1)
    handler.run_score_groups(2, None, None)
    assert epsilon_equal(handler.get_score_values('loss')['loss'], 2)

    warnings.simplefilter("ignore", DeprecationWarning)
    handler.push_score_values()
    handler.reset_score_values()
    assert np.isnan(handler.get_score_values('loss')['loss'])

    total_loss.append(1.5)
    assert epsilon_equal(handler.get_score_history('loss')['loss'], total_loss)


def test_ScoreGroupHook_epoch_3():
    handler.run_score_groups(5, None, None)
    assert epsilon_equal(handler.get_score_values('loss')['loss'], 5)

    warnings.simplefilter("ignore", DeprecationWarning)
    handler.push_score_values()
    handler.reset_score_values()
    assert np.isnan(handler.get_score_values('loss')['loss'])

    total_loss.append(5)
    assert epsilon_equal(handler.get_score_history('loss')['loss'], total_loss)
