import os
import torch
import typing
import warnings
import pydantic
import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader

from ..data.dataset import Dataset as NumpyDataset
from ..data.utils import TrainerDataLoader
from ..models.utils import TrainerModel
from ..utils.pydantic.types import DataLoaderType


class IterationArguments(pydantic.BaseModel):
    model: TrainerModel

    @typing.overload
    def is_training_hparams(self) -> bool:
        raise NotImplementedError

    @typing.overload
    def dict(self) -> typing.Dict[str, typing.Any]:
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

    def dict(self) -> typing.Dict[str, typing.Any]:
        fields = ('num_epochs', 'learning_rate',
                  'resume_epochs', 'num_epochs_per_validation',
                  'train_dl_batch_size', 'valid_dl_batch_size')
        return dict(zip(fields, map(self.__getattribute__, fields)))

    @property
    def train_dl_batch_size(self) -> int:
        return self.train_dl.batch_size

    @property
    def valid_dl_batch_size(self) -> typing.Optional[int]:
        if self.valid_dl is None:
            return None
        else:
            return self.valid_dl.batch_size

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

    def dict(self) -> typing.Dict[str, typing.Any]:
        fields = ('eval_dl_batch_size')
        return dict(zip(fields, map(self.__getattribute__, fields)))

    @property
    def eval_dl_batch_size(self):
        return self.eval_dl.batch_size


Hyperparameter = typing.NewType('Arguments', typing.Union[TrainingArguments,
                                                          EvaluatingArguments])
