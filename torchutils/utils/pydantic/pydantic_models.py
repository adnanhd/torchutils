import pydantic
import torch
import inspect
import warnings
from torch.nn import Module
from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchutils.metrics import MetricHandler
import torchsummary
from collections import OrderedDict
import typing
from .pydantic_types import (
    # NpScalarType,
    NpTorchType,
    ModuleType,
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


class TrainerModel(pydantic.BaseModel):
    model: ModuleType
    criterion: typing.Union[ModuleType, FunctionType]
    optimizer: OptimizerType
    scheduler: typing.Optional[SchedulerType]

    def __init__(
        self,
        model: Module,
        criterion: typing.Union[str, Module,
                                typing.Callable[[NpTorchType, NpTorchType],
                                                NpTorchType]],
        optimizer: typing.Union[str, Optimizer],
        scheduler: typing.Union[str, _LRScheduler, None] = None,
        ** kwargs,
    ):
        def obtain_registered_kwargs(fn: typing.Callable,
                                     kwargs: typing.Dict[str, typing.Any]):
            return dict(
                filter(
                    lambda item: item[0] in inspect.signature(
                        fn).parameters.keys(),
                    kwargs.items()
                )
            )

        if isinstance(criterion, str):
            if criterion in string_to_criterion_class:
                criterion_class = string_to_criterion_class[criterion]
            elif criterion in string_to_functionals:
                criterion_class = string_to_functionals[criterion]
            else:
                raise KeyError(
                    f"{criterion} is not a registered function or Module"
                )
            criterion_params = obtain_registered_kwargs(
                criterion_class, kwargs)
            criterion = criterion_class(**criterion_params)

        if isinstance(optimizer, str):
            if optimizer in string_to_optimizer_class:
                optimizer = string_to_optimizer_class[optimizer]
                params = obtain_registered_kwargs(optimizer, kwargs)
                optimizer = optimizer(model.parameters(), **params)
            else:
                raise KeyError(
                    f"{optimizer} is not a registered Optimizer"
                )

        if isinstance(scheduler, str):
            if scheduler in string_to_scheduler_class:
                scheduler = string_to_scheduler_class[scheduler]
                params = obtain_registered_kwargs(scheduler, kwargs)
                scheduler = scheduler(optimizer, **params)
            else:
                raise KeyError(
                    f"{scheduler} is not a registered Scheduler"
                )

        super(TrainerModel, self).__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler
        )

    def __setattr__(self, key, value):
        if key == 'device':
            self.model = self.model.to(device=value)
            return object.__setattr__(self.model, 'device', value)
        if key == 'dtype':
            self.model = self.model.to(dtype=value)
            return object.__setattr__(self.model, 'dtype', value)
        if key == 'learning_rate':
            self.optimizer.param_groups[0]['lr'] = value
            return object.__setattr__(self.optimizer, 'lr', value)
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

    def __getstate__(self) -> typing.Set[str]:
        if isinstance(self.criterion, Module):
            return {'model', 'optimizer', 'scheduler', 'criterion'}
        else:
            return {'model', 'optimizer', 'scheduler'}

    def state_dict(self):
        state = OrderedDict()
        for key in self.__getstate__():
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
        for key in self.__getstate__():
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

    def __init__(self, dataloader: DataLoader):
        super().__init__(dataloader=dataloader)

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
            self._hparams = TrainingArguments(**kwargs)
        else:
            self._hparams = EvaluatingArguments(**kwargs)
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
