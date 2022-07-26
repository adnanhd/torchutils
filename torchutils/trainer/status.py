import pydantic
import enum


class IterationStatus(pydantic.BaseModel):
    """ A class for holding information that """
    """changes as the trainer iterates """
    class StatusCode(enum.Enum):

        UNINITIALIZED = 6
        # An error occured before starting.
        # Started unsuccessfully.
        ABORTED = 1
        # Started successfully.
        STARTED = 2
        # Finished successfully.
        FINISHED = 0

        TRAINING_BATCH = 7
        TRAINING_EPOCH_FINISHED = 8

        VALIDATION = 9
        VALIDATION_RUN_FINISHED = 10

        EVALUATION = 11
        EVALUATION_RUN_FINISHED = 12

        # Training finishead early on purpose
        # StopTrainingError raised
        STOPPED = 3
        # @TODO: NOT IMPLEMENTED YET.
        CRUSHED = 4
        # An exception occured after starting,
        # i.e. finished unsuccessfully.
        FAILED = 5

    class Config:
        allow_mutation = True
    current_epoch: int = None
    current_batch: int = None
    _status_code: StatusCode = pydantic.PrivateAttr(
        default=StatusCode(StatusCode.UNINITIALIZED)
    )

    @property
    def status_code(self) -> int:
        return self._status_code.value

    @status_code.setter
    def status_code(self, new_code: StatusCode):
        return self.set_status_code(new_code)

    def set_status_code(self, status_code: StatusCode):
        assert isinstance(status_code, self.StatusCode)
        self._status_code = status_code

    @property
    def status_message(self) -> str:
        return self._status_code.name

    def __setattr__(self, name, value):
        if name == 'status_code':
            return object.__setattr__(self, 'status_code', value)
        return super().__setattr__(name, value)
