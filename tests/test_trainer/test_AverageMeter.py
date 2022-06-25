from torchutils.metrics import AverageMeter
from .utils import epsilon_equal

tracker = AverageMeter('Loss', fmt=':e')


def test_AverageMeter():
    tracker.reset()
    tracker.update(1e-2)
    tracker.update(1e-3)
    tracker.update(1e-4)
    assert epsilon_equal(tracker.average, (1e-2 + 1e-3 + 1e-4) / 3)

    tracker.reset()
    tracker.update(1)
    tracker.update(2)

    tracker.reset()
    tracker.update(5)
    assert epsilon_equal(tracker.average, 5)
