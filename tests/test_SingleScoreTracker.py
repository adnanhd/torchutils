from torchutils.utils.tracker import SingleScoreTracker

tracker = SingleScoreTracker()
# an acceptable error margin
epsilon = 1e-10


def test_5e2():
    tracker.update(1e-2)
    tracker.update(4e-2)
    tracker.update(1e-1)
    assert abs(tracker.average - 5e-2) < epsilon
