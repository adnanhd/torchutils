import pydantic
import enum


class IterationStatus(pydantic.BaseModel):
    """ A class for holding information that """
    """changes as the trainer iterates """
    class StatusCode(enum.Enum):
        # An error occured before starting.
        # Started unsuccessfully.
        ABORTED = 1
        # Started successfully.
        STARTED = 2
        # Finished successfully.
        FINISHED = 0
        # Training finishead early on purpose
        # StopTrainingError raised
        STOPPED = 3
        # @TODO: NOT IMPLEMENTED YET.
        CRUSHED = 4
        # An exception occured after starting,
        # i.e. finished unsuccessfully.
        FAILED = 5

        UNINITIALIZED = 6

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

    def set_status_code(self, status_code: StatusCode):
        assert isinstance(status_code, self.StatusCode)
        self._status_code = status_code

    @property
    def status_message(self) -> str:
        return self._status_code.name
