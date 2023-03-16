import inspect
import logging
import pydantic
import torch
import warnings
import torchsummary
import typing
import torch.nn as nn

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchutils.metrics import AverageScore, MetricHandler
from torchutils.utils import digest
from collections import OrderedDict
from torchutils.utils.pydantic.types import (
    # NpScalarType,
    NpTorchType,
    GradTensorType,
    ModuleType,
    OptimizerType,
    SchedulerType,
    CriterionType,
    # DatasetType
)
from torchutils.utils import (
    string_to_criterion_class,
    string_to_optimizer_class,
    string_to_scheduler_class,
    string_to_functionals,
    criterion_class_to_string,
    obtain_registered_kwargs
)


def _add_modules(self, modules, layer_num=None):
    if layer_num is None:
        for key, value in modules.items():
            self.add_module(key, value)
    else:
        for key, value in modules.items():
            self.add_module(f'{key}{layer_num}', value)


def _argument(*args, **kwargs):
    return {'args': args, 'kwargs': kwargs}


def _add_last_layer(self, output, choices=(None, ), *args, **kwargs):
    assert (output in choices) or isinstance(output, nn.Module)

    if output == 'linear':
        self.add_module('linear', nn.Linear(*args, **kwargs))
    elif output == 'dropout':
        self.add_module('dropout', nn.Dropout(*args, **kwargs))
    elif output == 'sigmoid':
        self.add_module('sigmoid', nn.Sigmoid(*args, **kwargs))
    elif isinstance(output, nn.Module):
        self.add_module('submodule', output)


def init_avg_loss():
    try:
        return AverageScore("Loss")
    except KeyError:
        avg = MetricHandler.MetricRegistrar.__score__['Loss']
        avg.reset()
        return avg


class TrainerModel(pydantic.BaseModel):
    model: ModuleType
    criterion: CriterionType
    optimizer: OptimizerType
    scheduler: typing.Optional[SchedulerType]
    arguments: typing.Dict = pydantic.Field(default_factory=dict)
    _loss: AverageScore = pydantic.PrivateAttr(default_factory=init_avg_loss)
    _logger: logging.Logger = pydantic.PrivateAttr(
        default_factory=logging.getLogger)
    _backward_hooks: typing.List[GradTensorType] = pydantic.PrivateAttr(
        default_factory=list)

    class Config:
        validate_assignment = True

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        if isinstance(value, TrainerModel):
            return value
        else:
            raise ValueError(f'{value} is not a {cls.__qualname__}')

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
        super().__init__(model=model, arguments=kwargs,
                         criterion=criterion, optimizer=optimizer,
                         scheduler=scheduler)
        self._logger = logging.getLogger(model.__class__.__qualname__)

    @classmethod
    def _validate_criterion(
        cls,
        criterion: typing.Union[nn.Module,
                                typing.Callable[[NpTorchType, NpTorchType],
                                                NpTorchType]],
        ** kwargs,
    ):
        if criterion in string_to_criterion_class.values():
            criterion_params = obtain_registered_kwargs(criterion, kwargs)
            criterion = criterion(**criterion_params)
        return criterion

    @classmethod
    def _validate_optimizer(
        cls,
        model: nn.Module,
        optimizer: Optimizer,
        ** kwargs,
    ):
        if optimizer in string_to_optimizer_class.values():
            params = obtain_registered_kwargs(optimizer, kwargs)
            optimizer = optimizer(model.parameters(), **params)
        return optimizer

    @classmethod
    def _validate_scheduler(
        cls,
        optimizer: Optimizer,
        scheduler: typing.Optional[_LRScheduler] = None,
        ** kwargs,
    ):
        if scheduler in string_to_scheduler_class.values():
            params = obtain_registered_kwargs(scheduler, kwargs)
            scheduler = scheduler(optimizer, **params)
        return scheduler

    @pydantic.root_validator
    def compile_all_modules(cls, modules):
        model = modules['model']
        kwargs = modules['arguments']
        criterion = modules['criterion']
        optimizer = modules['optimizer']
        scheduler = modules['scheduler']

        if 'verbose' in kwargs and kwargs['verbose']:
            print('compiling components...')

        if inspect.isclass(criterion):
            criterion = cls._validate_criterion(criterion, **kwargs)

        if inspect.isclass(optimizer):
            optimizer = cls._validate_optimizer(model, optimizer, **kwargs)
        else:
            try:
                params = obtain_registered_kwargs(optimizer.__class__, kwargs)
                optimizer.add_param_group(
                    {'params': model.parameters(), **params})
            except ValueError:
                # model params are already in the model
                pass  # i.e. model is not changed yet

        if inspect.isclass(scheduler):
            scheduler = cls._validate_scheduler(model, scheduler, **kwargs)
        elif scheduler is not None:
            scheduler.optimizer = optimizer

        return dict(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            arguments=kwargs,
        )

    def __setattr__(self, key, value):
        if key == 'device':
            self.model = self.model.to(device=value)
            return object.__setattr__(self.model, 'device', value)
        if key == 'dtype':
            self.model = self.model.to(dtype=value)
            return object.__setattr__(self.model, 'dtype', value)
        if key == 'learning_rate':
            param_groups = self.optimizer.param_groups
            for idx in range(param_groups.__len__()):
                param_groups[idx]['lr'] = value
            return object.__setattr__(self.optimizer, 'lr', value)
        return super().__setattr__(key, value)

    def to(self, dtype=None, device=None):
        kwargs = dict()
        if dtype is not None:
            kwargs['dtype'] = dtype
        if device is not None:
            kwargs['device'] = device
        self.model.to(**kwargs)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def learning_rate(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    @property
    def loss(self) -> float:
        return self._loss.average

    def reset_parameters(self):
        def fn(m: nn.Module):
            if isinstance(m, nn.Module) and hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        self.model.apply(fn)

    def init_parameters(self, fn):
        self.model.apply(fn)

    def __hash__(self) -> int:
        return int(digest(self.state_dict()), 16)

    @property
    def checksum(self) -> str:
        return digest(self.state_dict())

    def __call__(self, input):
        return self.model(input)

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
                warnings.warn(
                    f"{key} either absent in the state_dict or None."
                )
            elif hasattr(module, 'load_state_dict'):
                module.load_state_dict(state_dict[key])
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
            device=self.device.type)

    def train(self) -> None:
        self.model.train()
        if isinstance(self.criterion, Module):
            self.criterion.train()

    def eval(self) -> None:
        self.model.eval()
        if isinstance(self.criterion, Module):
            self.criterion.eval()

    def register_metrics_to(self, handler: MetricHandler):
        assert isinstance(handler, MetricHandler)
        handler.add_score_meters(self._loss)

    def scheduler_step(self):
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau):
            pass  # self.scheduler.step(self.loss)
        else:
            self.scheduler.step()

    def reset_backward(self):
        if self._backward_hooks.__len__() != 0:
            warnings.warn(
                "BackwardHook is not empty", RuntimeWarning
            )
            self._backward_hooks.clear()
        self._loss.reset()

    def _push_for_backward(self, tensor: torch.Tensor) -> None:
        if hasattr(tensor, 'requires_grad') and tensor.requires_grad:
            self._backward_hooks.append(tensor)

    def forward_pass(self, x, y, batch_idx=None):
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self._loss.update(loss.item())
        self._push_for_backward(loss)
        return y_pred

    def backward_pass(self):
        self.optimizer.zero_grad()
        if self._backward_hooks.__len__() == 0:
            warnings.warn(
                "TrainerModel.backward_pass receives no loss to backward"
                "check requires_grad attribute of input and output pairs",
                RuntimeWarning
            )
        while self._backward_hooks.__len__() != 0:
            self._backward_hooks.pop().backward()
        self.optimizer.step()
