from torchutils.logging import SlurmLogger, LoggingEvent
from pipeline import Pipeline


if __name__ == '__main__':
    epoch = SlurmLogger.getLogger(
        event=LoggingEvent.TRAINING_EPOCH,
        experiment='test_slurm'
    )
    batch = SlurmLogger.getLogger(
        event=LoggingEvent.TRAINING_BATCH,
        experiment='test_slurm'
    )
    Pipeline(
        epoch=epoch,
        batch=batch,
        sample=batch
    ).test_1()
