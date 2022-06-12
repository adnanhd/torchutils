from torchutils.utils.tracker import DataFrameRunHistory
from numpy import nan as NaN


run = DataFrameRunHistory()


def test_set_score_names():
    run.set_score_names(['loss', 'mse'])
    assert {'loss', 'mse'} == run.get_score_names()


def test_get_latest_score():
    run.set_latest_score('mse', 1e-5)
    run.set_latest_score('loss', 1e-2)
    assert run.get_latest_score('mse') == 1e-5
    assert run.get_latest_score('loss') == 1e-2


def test_get_score_values_mse():
    run._increment_epoch()
    run.set_latest_score('mse', 1e-2)
    assert run.get_score_values('mse') == [1e-5, 1e-2]


def test_get_score_values_loss():
    assert run.get_score_values('loss') == [1e-2, NaN]
