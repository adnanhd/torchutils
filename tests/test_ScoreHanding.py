from torchutils.utils.tracker import DataFrameRunHistory
from torchutils.utils.tracker import SingleScoreTracker
from collections import defaultdict


# initialize history and tracker
history = DataFrameRunHistory()
tracker = defaultdict(SingleScoreTracker)
# an acceptable error margin
epsilon = 1e-10


def on_initialization(score_names):
    history.set_score_names(score_names)
    tracker.clear()
    for score_name in score_names:
        tracker[score_name].reset()


def on_step_end(**scores):
    for score_name, score_value in scores.items():
        tracker[score_name].update(score_value)


def on_epoch_end(score_names):
    for score_name in score_names:
        history.set_latest_score(score_name, tracker[score_name].average)
        tracker[score_name].reset()


def test_iter1():
    score_names = ['mse', 'loss']
    on_initialization(score_names)
    on_step_end(mse=1e-2, loss=1e-2)
    on_step_end(mse=2e-2, loss=1e-2)
    on_step_end(mse=3e-2, loss=1e-2)
    on_epoch_end(score_names)
    assert abs(history.get_latest_score('mse') - 2e-2) < epsilon
    assert abs(history.get_latest_score('loss') - 1e-2) < epsilon


def test_iter2():
    score_names = ['loss']
    on_initialization(score_names)
    on_step_end(loss=1e-2)
    on_step_end(loss=4e-2)
    on_step_end(loss=1e-1)
    on_epoch_end(score_names)
    assert abs(history.get_latest_score('loss') - 5e-2) < epsilon
