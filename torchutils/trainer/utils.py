import pydantic
import os
import numpy as np
import warnings
import enum
import torch
import typing
from torch.utils.data.dataset import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader
from ..data.dataset import Dataset as NumpyDataset
from ..metrics.handler import MetricHandler
from ..metrics.history import RunHistory
from ..data.utils import TrainerDataLoader
from ..models.utils import TrainerModel
from ..utils.pydantic.types import NpTorchType, DataLoaderType


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


class IterationArguments(pydantic.BaseModel):
    model: TrainerModel

    @typing.overload
    def is_training_hparams(self) -> bool:
        raise NotImplementedError

    def create_dataloader(
        self,
        dataset: TorchDataset,
        train_mode: bool = True,
        batch_size: typing.Optional[int] = None,
        **kwargs,
    ) -> DataLoader:
        assert isinstance(dataset, TorchDataset), \
            "Dataset must be inherited from {TorchDataset}"

        try:
            if isinstance(dataset.x, np.ndarray) \
                    and self.device != torch.device('cpu'):
                dataset.x = torch.from_numpy(dataset.x)
            dataset.x = dataset.x.to(
                device=self.device,
                dtype=self.xtype
            )
            if isinstance(dataset.y, np.ndarray) and \
                    self.device != torch.device('cpu'):
                dataset.y = torch.from_numpy(dataset.y)
            dataset.y = dataset.y.to(
                device=self.device,
                dtype=self.ytype
            )
        except AttributeError:
            warnings.warn(
                "Using a Dataset not derived from torchutils.data.Dataset "
                "is dangerous for dtype integrity"
            )

        kwargs.setdefault('shuffle', train_mode)
        kwargs.setdefault('pin_memory', not torch.cuda.is_available())
        kwargs.setdefault(
            'num_workers', 0 if torch.cuda.is_available() else os.cpu_count())
        if not train_mode or batch_size is None:
            kwargs['batch_size'] = dataset.__len__()
        else:
            kwargs['batch_size'] = batch_size

        if isinstance(dataset, NumpyDataset):
            return dataset.dataloader(**kwargs)
        else:
            return DataLoader(dataset, **kwargs)


class TrainingArguments(IterationArguments):
    class Config:
        allow_mutation = False

    num_epochs: int = pydantic.Field(ge=0, description="Number of epochs")
    learning_rate: float = pydantic.Field(ge=0.0, le=1.0)
    resume_epochs: int = 0
    num_epochs_per_validation: int = 1

    train_dl: typing.Union[TrainerDataLoader, DataLoaderType]
    valid_dl: typing.Optional[typing.Union[TrainerDataLoader,
                                           DataLoaderType]] = None

    @property
    def is_training_hparams(self) -> bool:
        return True

    @property
    def train_dl_batch_size(self) -> int:
        return self._train_dl.batch_size

    @property
    def valid_dl_batch_size(self) -> typing.Optional[int]:
        if self._valid_dl is None:
            return None
        else:
            return self._valid_dl.batch_size

    @pydantic.validator('train_dl')
    @classmethod
    def validate_train_dataloader(
            cls, field_type
    ) -> TrainerDataLoader:
        if isinstance(field_type, TrainerDataLoader):
            return field_type
        elif isinstance(field_type, torch.utils.data.DataLoader):
            return TrainerDataLoader(dataloader=field_type)
        else:
            raise ValueError(
                f'Not possible to accept a type of {type(field_type)}')

    @pydantic.validator('valid_dl')
    @classmethod
    def validate_valid_dataloader(
            cls, field_type) -> typing.Optional[TrainerDataLoader]:
        if field_type is None:
            return field_type
        else:
            return cls.validate_train_dataloader(field_type)


class EvaluatingArguments(IterationArguments):
    class Config:
        allow_mutation = False

    eval_dl: typing.Union[TrainerDataLoader, DataLoaderType]

    @pydantic.validator('eval_dl')
    @classmethod
    def validate_eval_dataloader(
            cls, field_type
    ) -> TrainerDataLoader:
        return TrainingArguments.validate_train_dataloader(field_type)

    @property
    def is_training_hparams(self) -> bool:
        return False

    @property
    def eval_dl_batch_size(self):
        return self._eval_dl.batch_size


class IterationBatch(pydantic.BaseModel):
    # @TODO: create validator for those types
    # xtype: str
    # ytype: str
    # device: str
    input: typing.Optional[NpTorchType]
    preds: typing.Optional[NpTorchType]
    target: typing.Optional[NpTorchType]

    class Config:
        allow_mutation = False

    def collate_fn(self, input, preds, target):
        self.Config.allow_mutation = True
        self.input = input
        self.target = target
        self.preds = preds
        self.Config.allow_mutation = False


class IterationInterface(pydantic.BaseModel):
    batch: IterationBatch = IterationBatch()
    status: IterationStatus = IterationStatus()
    hparams: IterationArguments = pydantic.Field(None)

    _at_epoch_end: bool = pydantic.PrivateAttr(False)
    _metric_handler: MetricHandler = pydantic.PrivateAttr()
    _metric_history: RunHistory = pydantic.PrivateAttr()
    _score_names: typing.Set[str] = pydantic.PrivateAttr()

    def __init__(self,
                 metrics: MetricHandler,
                 history: RunHistory,
                 hparams: IterationArguments):
        super().__init__()
        self.hparams = hparams
        self._metric_handler = metrics
        self._metric_history = history
        score_names = self._metric_history.get_score_names()
        self._score_names = score_names

    # functions for IterationHandler
    def collate_fn(self, input, preds, target):
        self._at_epoch_end = False
        self.batch.collate_fn(input, preds, target)
        self._metric_handler.run_score_functional(preds=preds,
                                                  target=target)

    def set_metric_scores(self):
        self._at_epoch_end = True

    def reset_metric_scores(self):
        score_values = self._metric_handler.get_score_averages(
            *self._metric_history.get_score_names())
        for score_name, score_value in score_values.items():
            self._metric_history.set_latest_score(score_name, score_value)
        # @TODO: instead of (postincrementing) _increment_epoch
        # use allocate_score_values (preincrementing)
        self._metric_history._increment_epoch()
        self._metric_handler.reset_score_values()

    # functions for Callbacks and Third Party Comp.

    def get_current_scores(self,
                           *score_names: str
                           ) -> typing.Dict[str, float]:
        """ Returns the latest step or epoch values, depending on
        whether it has finished itereting over the current epoch or not """
        # @TODO: make calls without star
        if self._at_epoch_end:
            return self._metric_handler.get_score_averages(*score_names)
        else:
            return self._metric_handler.get_score_values(*score_names)

    def get_stored_scores(
            self,
            *score_names: str
    ) -> typing.Dict[str, typing.List[float]]:
        """ Returns the all epoch values with given score names """
        return {
            score_name: self._metric_history.get_score_values(score_name)
            for score_name in score_names
        }
