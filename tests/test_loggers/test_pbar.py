from torchutils.logging.pbar import BatchProgressBar, EpochProgressBar, SampleProgressBar
from pipeline import Pipeline


if __name__ == '__main__':
    Pipeline(
        epoch=EpochProgressBar(position=1, leave=True),
        batch=BatchProgressBar(position=0, leave=False),
        sample=SampleProgressBar(position=0, leave=False)
    ).test_1()
