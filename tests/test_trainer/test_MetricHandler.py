from torchutils.metrics import AverageMeter, MetricHandler
from .utils import epsilon_equal
import numpy as np
import warnings

handler = MetricHandler()
handler.add_score_meters(AverageMeter('Loss', fmt=':e'))
handler.init_score_history('loss')
# initialize the score value with the first epoch
total = [(1e-2 + 1e-3 + 1e-4) / 3]


def test_MetricHandler_test1():
    assert not handler._history._has_latest_row
    assert handler._history.current_epoch == 1

    handler.set_scores_values(loss=1e-2)
    assert epsilon_equal(handler.get_score_values('loss')['loss'], 1e-2)
    handler.set_scores_values(loss=1e-3)
    assert epsilon_equal(handler.get_score_values('loss')['loss'], 1e-3)
    handler.set_scores_values(loss=1e-4)
    assert epsilon_equal(handler.get_score_values('loss')['loss'], 1e-4)

    warnings.simplefilter("ignore", DeprecationWarning)
    handler.push_score_values()
    handler.reset_score_values()
    assert np.isnan(handler.get_score_values('loss')['loss'])
    assert epsilon_equal(handler.get_score_history('loss')['loss'], total)


def test_MetricHandler_test2():
    assert not handler._history._has_latest_row
    assert handler._history.current_epoch == 2

    handler.set_scores_values(loss=1)
    assert epsilon_equal(handler.get_score_values('loss')['loss'], 1)
    handler.set_scores_values(loss=2)
    assert epsilon_equal(handler.get_score_values('loss')['loss'], 2)

    warnings.simplefilter("ignore", DeprecationWarning)
    handler.push_score_values()
    handler.reset_score_values()
    assert np.isnan(handler.get_score_values('loss')['loss'])
    total.append(1.5)  # add the score value of the current epoch
    assert epsilon_equal(handler.get_score_history('loss')['loss'], total)


def test_MetricHandler_test3():
    assert handler._history.current_epoch == 3
    assert not handler._history._has_latest_row

    handler.set_scores_values(loss=5)
    assert epsilon_equal(handler.get_score_values('loss')['loss'], 5)

    warnings.simplefilter("ignore", DeprecationWarning)
    handler.push_score_values()
    handler.reset_score_values()
    assert np.isnan(handler.get_score_values('loss')['loss'])
    total.append(5.0)  # add the score value of the current epoch
    assert epsilon_equal(handler.get_score_history('loss')['loss'], total)
