import pydantic
import enum
import torch
import typing
from torch.nn import Module
from torchutils.metrics import MetricHandler
from torch.utils.data.dataloader import DataLoader
from torchutils.data.utils import TrainerDataLoader
from torchutils.models.utils import TrainerModel
from torchutils.utils.pydantic.types import (
    NpTorchType,
    DataLoaderType,
)


class TrainingArguments(pydantic.BaseModel):
    class Config:
        allow_mutation = False

    num_epochs: int = pydantic.Field(ge=0, description="Number of epochs")
    learning_rate: float = pydantic.Field(ge=0.0, le=1.0)
    resume_epochs: int = 0
    train_dl_batch_size: int
    valid_dl_batch_size: typing.Optional[int] = -1  # All elements at a batch
    num_epochs_per_validation: int = 1


class EvaluatingArguments(pydantic.BaseModel):
    class Config:
        allow_mutation = False
    eval_dl_batch_size: int = 1  # One element per patch

    def num_steps(self):
        return self.dataloader.__len__()


class TrainerStatus(pydantic.BaseModel):
    class StatusCode(enum.Enum):
        """
        STARTED -- started successfully
        FINISHED -- finished successfully
        ABORTED -- an exception occurred
        STOPPED -- StopTrainingError occured
        ----
        FAILED
        CRUSHED
        """
        FINISHED_SUCCESSFULLY = 0
        UNINITIALIZED = 1
        STARTED_SUCCESSFULLY = 2
        STOP_TRAINING_ERROR_OCCURED = 3
        AN_EXTERNAL_ERROR_OCCURED = 4

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


class HandlerArguments(pydantic.BaseModel):
    model: TrainerModel
    _status_ptr: typing.List[TrainerStatus] = pydantic.PrivateAttr(
        [TrainerStatus()])
    _hparams: typing.Union[TrainingArguments,
                           EvaluatingArguments] = pydantic.PrivateAttr(None)
    train_dl: typing.Optional[typing.Union[TrainerDataLoader,
                                           DataLoaderType]] = None
    valid_dl: typing.Optional[typing.Union[TrainerDataLoader,
                                           DataLoaderType]] = None
    eval_dl: typing.Optional[typing.Union[TrainerDataLoader,
                                          DataLoaderType]] = None

    def __init__(self,
                 model: Module,
                 train_dl: DataLoader = None,
                 valid_dl: DataLoader = None,
                 eval_dl: DataLoader = None,
                 **kwargs):
        is_in_training = train_dl is not None and eval_dl is None
        is_in_evaluating = train_dl is None and eval_dl is not None
        assert is_in_evaluating != is_in_training
        dataloaders = {
            'train_dl': train_dl,
            'valid_dl': valid_dl,
            'eval_dl': eval_dl
        }
        dataloaders = {
            dl_name: TrainerDataLoader(dataloader=dataloader)
            if isinstance(dataloader, DataLoader) else dataloader
            for dl_name, dataloader in dataloaders.items()
            if dataloader is not None
        }
        super().__init__(model=model, **dataloaders)
        if is_in_training:
            if valid_dl is not None:
                kwargs['valid_dl_batch_size'] = valid_dl.batch_size
            else:
                kwargs['valid_dl_batch_size'] = None
            self._hparams = TrainingArguments(
                learning_rate=model.learning_rate,
                train_dl_batch_size=train_dl.batch_size,
                **kwargs
            )
        else:
            self._hparams = EvaluatingArguments(
                eval_dl_batch_size=eval_dl.batch_size,
                **kwargs
            )
        self._status_ptr = [TrainerStatus()]

    @property
    def status(self) -> TrainerStatus:
        return self._status_ptr[0]

    @property
    def hparams(self) -> typing.Union[TrainingArguments,
                                      EvaluatingArguments]:
        return self._hparams

    def set_status(self, batch=None, epoch=None) -> None:
        if batch is not None:
            self._status_ptr[0].current_batch = batch
        if epoch is not None:
            self._status_ptr[0].current_epoch = epoch

    def reset_status(self) -> None:
        self._status_ptr[0].current_epoch = None
        self._status_ptr[0].current_batch = None

    class Config:
        allow_mutation = True

    @pydantic.validator('train_dl', 'valid_dl', 'eval_dl')
    @classmethod
    def validate_dataloaders(cls, field_type):
        if isinstance(field_type, TrainerDataLoader):
            return field_type
        elif isinstance(field_type, torch.utils.data.DataLoader):
            return TrainerDataLoader(dataloader=field_type)
        else:
            raise ValueError(
                f'Not possible to accept a type of {type(field_type)}')

    @property
    def istrainable(self):
        if isinstance(self.args, TrainerDataLoader):
            return True
        elif isinstance(self.args, EvaluatingArguments):
            return False
        else:
            return None


class CurrentIterationStatus(pydantic.BaseModel):
    _at_epoch_end: bool = pydantic.PrivateAttr(False)
    _metric_handler: MetricHandler = pydantic.PrivateAttr()
    _epoch_idx: typing.Optional[int] = pydantic.PrivateAttr(None)
    _batch_idx: typing.Optional[int] = pydantic.PrivateAttr(None)
    _x: typing.Optional[NpTorchType] = pydantic.PrivateAttr(None)
    _y_true: typing.Optional[NpTorchType] = pydantic.PrivateAttr(None)
    _y_pred: typing.Optional[NpTorchType] = pydantic.PrivateAttr(None)
    _status_ptr: typing.List[TrainerStatus] = pydantic.PrivateAttr([None])

    def __init__(self, handler: MetricHandler):
        super().__init__()
        self._metric_handler = handler
        self._status_ptr = [None]

    @property
    def x(self):
        return self._x

    @property
    def y_pred(self):
        return self._y_pred

    @property
    def y_true(self):
        return self._y_true

    @property
    def status(self) -> typing.Optional[TrainerStatus]:
        return self._status_ptr[0]

    def __setattr__(self, key, value):
        if key == 'status':
            assert isinstance(value, TrainerStatus)
            self._status_ptr[0] = value
            return object.__setattr__(self._status_ptr[0], 'status', value)
        return super().__setattr__(key, value)

    def get_score_names(self):
        return self._metric_handler.get_score_names()

    def set_score_names(self, score_names: typing.Iterable[str]):
        self._metric_handler.set_score_names(score_names)

    def reset_score_names(self):
        self._metric_handler.set_score_names()

    # every step end
    def set_current_scores(self, x, y_true, y_pred, batch_idx=None):
        """ Sets the current step input and outputs and calls score groups """
        self._x = x
        self._y_true = y_true
        self._y_pred = y_pred
        self._at_epoch_end = False
        self._metric_handler.run_score_functional(preds=y_pred, target=y_true)
        self._batch_idx = batch_idx

    # every epoch end
    def average_current_scores(self, epoch_idx=None):
        """ Pushes the scores values of the current epoch to the history
        in the metric handler and clears the score values of the all steps
        in the latest epoch in the metric handler """
        self._at_epoch_end = True
        self._metric_handler.push_score_values()
        self._metric_handler.reset_score_values()
        self._epoch_idx = epoch_idx

    def get_current_scores(self, *score_names: str) -> typing.Dict[str, float]:
        """ Returns the latest step or epoch values, depending on
        whether it has finished itereting over the current epoch or not """
        if len(score_names) == 0:
            score_names = self._metric_handler.get_score_names()
        if self._at_epoch_end:
            return self._metric_handler.seek_score_history(*score_names)
        else:
            return self._metric_handler.get_score_values(*score_names)

    def get_score_history(
            self,
            *score_names: str
    ) -> typing.Dict[str, typing.List[float]]:
        """ Returns the all epoch values with given score names """
        if len(score_names) == 0:
            score_names = self.get_score_names()
        return self._metric_handler.get_score_history(*score_names)
