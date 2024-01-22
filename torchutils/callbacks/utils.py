import typing
import logging
import enum


class LogLevelEnum(enum.Enum):
    TRAINING_STEP_END: typing.ClassVar[int] = 15
    TRAINING_EPOCH_END: typing.ClassVar[int] = 25
    VALIDATION_STEP_END: typing.ClassVar[int] = 16
    VALIDATION_END: typing.ClassVar[int] = 26
    EVALUATION_STEP_END: typing.ClassVar[int] = 17
    EVALUATION_END: typing.ClassVar[int] = 27


for level in LogLevelEnum.__members__.values():
    logging.addLevelName(level=level.value, levelName=level.name)

class GoalEnum(enum.Enum):
    minimize = 'minimize'
    maximize = 'maximize'