import pydantic
import torch
import os
import warnings
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchutils.metrics import MetricHandler, AverageMeter
import torchsummary
from collections import OrderedDict
import typing
import torch.optim as optimizers
from .pydantic_types import (
    # NpScalarType,
    NpTorchType,
    ModuleType,
    LossType,
    OptimizerType,
    SchedulerType,
    DataLoaderType,
    FunctionType,
    # DatasetType
)
from torchutils.utils import (
    string_to_criterion_class,
    string_to_optimizer_class,
    string_to_scheduler_class,
    string_to_functionals
)


class TrainerModelBuilder(pydantic.BaseModel):
    model: ModuleType
    critr: typing.Union[str, ModuleType, FunctionType]
    optim: typing.Union[str, OptimizerType]
    sched: typing.Optional[typing.Union[str, SchedulerType]]
    _criterion_kwargs: typing.Optional[typing.Dict] = pydantic.PrivateAttr({})
    _optimizer_kwargs: typing.Optional[typing.Dict] = pydantic.PrivateAttr({})
    _scheduler_kwargs: typing.Optional[typing.Dict] = pydantic.PrivateAttr({})

    @pydantic.validator('critr')
    @classmethod
    def validate_criterion(
        cls, criterion: typing.Any
    ) -> typing.Union[str, Module, callable]:
        if isinstance(criterion, Module) or callable(criterion):
            return criterion
        elif criterion not in string_to_criterion_class \
                and criterion not in string_to_functionals:
            raise ValueError(
                f"{criterion} is not valid criterion class name"
            )
        else:
            return criterion

    @pydantic.validator('optim')
    @classmethod
    def validate_optimizer(
        cls, optimizer: typing.Any
    ) -> typing.Union[str, Optimizer]:
        if isinstance(optimizer, Optimizer):
            return optimizer
        elif optimizer not in string_to_optimizer_class:
            raise ValueError(f"{optimizer} is not valid optimizer class name")
        else:
            return optimizer

    @pydantic.validator('sched')
    @classmethod
    def validate_scheduler(
            cls, scheduler: typing.Any
    ) -> typing.Union[str, _LRScheduler]:
        if isinstance(scheduler, _LRScheduler):
            return scheduler
        elif scheduler not in string_to_scheduler_class:
            raise ValueError(f"{scheduler} is not valid scheduler class name")
        else:
            return scheduler

    def configure_optimizer(self, lr: float, **kwargs) -> Optimizer:
        if isinstance(self.optim, str):
            optimizer_class = string_to_optimizer_class[self.optim]
            return optimizer_class(self.model.parameters(), lr=lr, **kwargs)
        else:
            return self.optim

    def configure_scheduler(self, **kwargs) -> typing.Optional[_LRScheduler]:
        if isinstance(self.sched, str):
            scheduler_class = string_to_scheduler_class[self.sched]
            return scheduler_class(self.optimizer, **kwargs)
        else:
            return self.sched

    def configure_criterion(self, **kwargs) -> Module:
        if isinstance(self.critr, str):
            criterion_class = string_to_criterion_class[self.critr]
            return criterion_class(**kwargs)
        else:
            return self.critr

    def set_criterion_arguments(self, **kwargs):
        self._criterion_kwargs.update(kwargs)

    def set_optimizer_arguments(self, **kwargs):
        self._optimizer_kwargs.update(kwargs)

    def set_scheduler_arguments(self, **kwargs):
        self._scheduler_kwargs.update(kwargs)

    def reset_criterion_arguments(self):
        self._criterion_kwargs.clear()

    def reset_optimizer_arguments(self):
        self._optimizer_kwargs.clear()

    def reset_scheduler_arguments(self):
        self._scheduler_kwargs.clear()

    def compile(self, lr: float):
        return TrainerModel(
            model=self.model,
            criterion=self.configure_criterion(**self._criterion_kwargs),
            optimizer=self.configure_optimizer(lr, **self._optimizer_kwargs),
            scheduler=self.configure_scheduler(**self._scheduler_kwargs)
        )


class TrainerModel(pydantic.BaseModel):
    model: ModuleType
    criterion: typing.Union[ModuleType, FunctionType]
    optimizer: OptimizerType
    scheduler: typing.Optional[SchedulerType]

    def __setattr__(self, key, value):
        if key == 'device':
            self.model = self.model.to(device=value)
            return object.__setattr__(self.model, 'device', value)
        if key == 'dtype':
            self.model = self.model.to(dtype=value)
            return object.__setattr__(self.model, 'dtype', value)
        return super().__setattr__(key, value)

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def _get_attr_names(self) -> typing.Set[str]:
        if isinstance(self.criterion, Module):
            return {'model', 'optimizer', 'scheduler', 'criterion'}
        else:
            return {'model', 'optimizer', 'scheduler'}

    def state_dict(self):
        state = OrderedDict()
        for key in self._get_attr_names():
            module = self.__getattribute__(key)
            if module is None:
                continue
            elif hasattr(module, 'state_dict'):
                state[key] = module.state_dict()
            else:
                warnings.warn(
                    f"{key} has no state_dict() attribute.", RuntimeWarning
                )
        return state

    def load_state_dict(
            self,
            state_dict: typing.Dict[str, "torch.Tensor"]) -> None:
        for key in self._get_attr_names():
            module = getattr(self, key)
            if module is None or key not in state_dict:
                continue
            elif hasattr(module, 'load_state_dict'):
                module.load_state_dict(state_dict.pop(key))
            else:
                warnings.warn(
                    f"{key} exits in the state_dict but that "
                    f"of {self.__qualname__} has no load_state_dict()"
                    "attribute.", RuntimeWarning
                )

    def summary(self, input_size, batch_size=-1) -> None:
        torchsummary.summary(
            self.model, input_size,
            batch_size=batch_size,
            device=self.device)

    def train(self) -> None:
        self.model.train()
        self.criterion.train()

    def eval(self) -> None:
        self.model.eval()
        self.criterion.eval()

    def optimizer_step(self):
        self.optimizer.step()

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def scheduler_step(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def __call__(self, input):
        return self.model(input)

    def forward_pass(self, x, y, batch_idx=None):
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        return y_pred, loss

    def compile(
            self,
            model: Module = None,
            loss: typing.Union[Module, callable] = None,
            optimizer: Optimizer = None,
            scheduler: _LRScheduler = None):
        if model is not None:
            self.model = model
        if loss is not None:
            self.criterion = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if scheduler is not None:
            self.scheduler = scheduler


class TrainingArguments(pydantic.BaseModel):
    class Config:
        allow_mutation = False

    num_epochs: int = pydantic.Field(ge=0, description="Number of epochs")
    learning_rate: float = pydantic.Field(ge=0.0, le=1.0)
    resume_epochs: int = 0
    train_dl_batch_size: int
    valid_dl_batch_size: int = -1  # All elements at a batch
    num_epochs_per_validation: int = 1


class EvaluatingArguments(pydantic.BaseModel):
    class Config:
        allow_mutation = False
    eval_dl_batch_size: int = 1  # One element per patch


class TrainerDataLoader(pydantic.BaseModel):
    class Config:
        allow_mutation = False
    dataloader: DataLoaderType

    @property
    def dataset(self):
        return self.dataloader.dataset

    @property
    def batch_size(self):
        return self.dataloader.batch_size

    @property
    def num_steps(self):
        return self.dataloader.__len__()


class TrainerStatus(pydantic.BaseModel):
    class Config:
        allow_mutation = True
    current_epoch: int = None
    current_batch: int = None

    def lock(self):
        if self.__config__.allow_mutation:
            self.__config__.allow_mutation = False

            def unlock(self):
                self.__config__.allow_mutation = True
            return unlock
        raise LookupError("HandlerArguments has been locked already.")


class CurrentIterationStatus(pydantic.BaseModel):
    _metric_handler: MetricHandler = pydantic.PrivateAttr()
    _x: typing.Optional[NpTorchType] = pydantic.PrivateAttr(None)
    _y_true: typing.Optional[NpTorchType] = pydantic.PrivateAttr(None)
    _y_pred: typing.Optional[NpTorchType] = pydantic.PrivateAttr(None)
    _epoch_idx: typing.Optional[int] = pydantic.PrivateAttr(None)
    _batch_idx: typing.Optional[int] = pydantic.PrivateAttr(None)
    _at_epoch_end: bool = pydantic.PrivateAttr(False)

    def __init__(self, handler: MetricHandler):
        super().__init__()
        self._metric_handler = handler

    # @property
    # def current_epoch(self):
    #     return self._epoch_idx

    # @property
    # def current_batch(self):
    #     return self._batch_idx

    @property
    def x(self):
        return self._x

    @property
    def y_pred(self):
        return self._y_pred

    @property
    def y_true(self):
        return self._y_true

    def get_score_names(self):
        return self._metric_handler.get_score_names()

    # every step end
    def set_current_scores(self, x, y_true, y_pred, batch_idx=None):
        """ Sets the current step input and outputs and calls score groups """
        self._x = x
        self._y_true = y_true
        self._y_pred = y_pred
        self._at_epoch_end = False
        self._metric_handler.run_score_groups(x, y_true, y_pred)
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

    def get_current_scores(self, *score_names: str
                           ) -> typing.Dict[str, float]:
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


class HandlerArguments(pydantic.BaseModel):
    args: typing.Union[TrainingArguments, EvaluatingArguments] = None
    model: TrainerModel
    status_ptr: typing.List[TrainerStatus]
    train_dl: typing.Optional[TrainerDataLoader] = None
    valid_dl: typing.Optional[TrainerDataLoader] = None
    eval_dl: typing.Optional[TrainerDataLoader] = None

    @property
    def status(self):
        return self.status_ptr[0]

    def update_status(self, batch=None, epoch=None):
        if batch is not None:
            self.status_ptr[0].current_batch = batch
        if epoch is not None:
            self.status_ptr[0].current_epoch = epoch

    def reset_status(self):
        self.status_ptr[0].current_epoch = None
        self.status_ptr[0].current_batch = None

    class Config:
        allow_mutation = True

    def set_arguments(self):
        if self.__config__.allow_mutation:
            self.__config__.allow_mutation = False

            def arguments_setter_callback(
                    new_args: typing.Union[TrainingArguments,
                                           EvaluatingArguments],
                    dataloaders: typing.Dict[str,
                                             typing.Optional[TrainerDataLoader]
                                             ] = dict()):
                self.__config__.allow_mutation = True
                self.args = new_args
                for dl_name in ('train_dl', 'valid_dl', 'eval_dl'):
                    dl = dataloaders.setdefault(dl_name, None)
                    if dl is not None:
                        dl = TrainerDataLoader(dataloader=dl)
                        self.__setattr__(dl_name, dl)
                self.__config__.allow_mutation = False
            return arguments_setter_callback
        raise LookupError("HandlerArguments has been locked already.")

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
    def is_train(self):
        if isinstance(self.args, TrainerDataLoader):
            return True
        elif isinstance(self.args, EvaluatingArguments):
            return False
        else:
            return None
