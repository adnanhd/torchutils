from pipeline import Pipeline
from torchutils.logging.wandb import WandbLogger
from torchutils.logging.nop import NoneLogger


if __name__ == '__main__':
    Pipeline(
        epoch=WandbLogger.getLogger(None),
        batch=NoneLogger(),
        sample=NoneLogger()
    ).test_1()
