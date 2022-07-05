import pydantic
import torch
import warnings
import torchsummary
import typing
import torch.nn as nn

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchutils.metrics import AverageMeter, MetricHandler
from collections import OrderedDict
from torchutils.utils.pydantic.types import (
    # NpScalarType,
    NpTorchType,
    GradTensorType,
    ModuleType,
    OptimizerType,
    SchedulerType,
    FunctionType,
    # DatasetType
)
from torchutils.utils import (
    string_to_criterion_class,
    string_to_optimizer_class,
    string_to_scheduler_class,
    string_to_functionals,
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
    return AverageMeter("Loss")


class TrainerModel(pydantic.BaseModel):
    model: ModuleType
    criterion: typing.Union[ModuleType, FunctionType]
    optimizer: OptimizerType
    scheduler: typing.Optional[SchedulerType]
    _loss: AverageMeter = pydantic.PrivateAttr(default_factory=init_avg_loss)
    _backward_hooks: typing.List[GradTensorType] = pydantic.PrivateAttr(
        default_factory=list)

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

        super().__init__(model=model, criterion=criterion, **kwargs,
                         optimizer=optimizer, scheduler=scheduler)

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
            self.scheduler.step(self.loss)
        else:
            self.scheduler.step()

    def reset_backward(self):
        self.scheduler_step()
        if self._backward_hooks.__len__() != 0:
            warnings.warn(
                "BackwardHook is not empty", RuntimeWarning
            )
            self._backward_hooks.clear()

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
