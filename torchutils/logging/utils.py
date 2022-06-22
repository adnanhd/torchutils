import enum


class LoggerMethodNotImplError(Exception):
    pass


class LoggingEvent(enum.Enum):
    TRAINING_BATCH = 0
    TRAINING_EPOCH = 1
    VALIDATION_RUN = 2
    EVALUATION_RUN = 3
