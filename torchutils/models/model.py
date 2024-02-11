import logging
import inspect
import pydantic
import typing
import torch
from functools import reduce
from collections import OrderedDict

from ..utils import digest
from .._dev_utils import AverageMeter, TrainerMeterModel
from .tensor import GradTensor, Tensor
from .builders import (
    NeuralNet,
    Criterion,
    Functional,
    Optimizer,
    Scheduler,
)


class TrainerModel(TrainerMeterModel):
    arguments: typing.Dict = pydantic.Field(default_factory=dict)
    criterion: typing.Union[Criterion, Functional]
    model: NeuralNet
    optimizer: Optimizer
    scheduler: typing.Optional[Scheduler]
    _backward_hooks: typing.List[GradTensor] = pydantic.PrivateAttr(default_factory=list)
    _loss: AverageMeter = pydantic.PrivateAttr()

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
        model: NeuralNet,
        criterion: Criterion,
        optimizer: Optimizer,
        modelname: str = None,
        scheduler: typing.Optional[Scheduler] = None,
        scores: typing.Set[str] = set(),
        ** kwargs,
    ):
        if modelname is None:
            modelname = model.__class__.__qualname__
        super().__init__(model=model, arguments=kwargs, 
                         criterion=criterion, optimizer=optimizer,
                         scheduler=scheduler, scores=scores)
        if inspect.isfunction(self.criterion):
            lossname = self.criterion.__name__
        else:
            lossname = self.criterion.__class__.__name__
        self._loss = AverageMeter(name=lossname)
        self._buffer['lossname'] = lossname
        self._buffer['modelname'] = modelname

    @pydantic.computed_field
    def criterion_name(self) -> str:
        return self._buffer['lossname']

    @pydantic.computed_field
    def num_parameters(self) -> int:
        return reduce(lambda acc, param: acc.__add__(param.numel()), self.model.parameters(), 0)

    # STATE DICT FUNCTIONS
    def __state_names__(self) -> typing.Set[str]:
        if isinstance(self.criterion, torch.nn.Module):
            return {'model', 'optimizer', 'scheduler', 'criterion'}
        else:
            return {'model', 'optimizer', 'scheduler'}

    def state_dict(self):
        state = OrderedDict()
        for key in self.__state_names__():
            module = self.__getattribute__(key)
            if module is None:
                continue
            elif hasattr(module, 'state_dict'):
                state[key] = module.state_dict()
            else:
                self.log_warn(
                    f"{key} has no state_dict() attribute."
                )
        return state

    def load_state_dict(
            self,
            state_dict: typing.Dict[str, "torch.Tensor"]) -> None:
        for key in self.__state_names__():
            module = getattr(self, key)
            if module is None or key not in state_dict:
                self.log_warn(
                    f"{key} either absent in the state_dict or None."
                )
            elif hasattr(module, 'load_state_dict'):
                module.load_state_dict(state_dict[key])
            else:
                self.log_warn(
                    f"{key} exits in the state_dict but that "
                    f"of {self.__qualname__} has no load_state_dict()"
                    "attribute."
                )

    def __hash__(self) -> int:
        return int(digest(self.state_dict()), 16)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @pydantic.computed_field
    def checksum(self) -> str:
        return digest(self.state_dict())

    def summary(self, input_size, batch_size=-1) -> None:
        try:
            import torchsummary
            torchsummary.summary(
                self.model, input_size,
                batch_size=batch_size,
                device=self.device.type)
        except ImportError:
            self.log_warn("torchsummary not installed")
            print("torchsummary not installed")

    # TRAINER FUNCTIONS
    def train(self) -> None:
        self.model.train()
        if isinstance(self.criterion, torch.nn.Module):
            self.criterion.train()

    def eval(self) -> None:
        self.model.eval()
        if isinstance(self.criterion, torch.nn.Module):
            self.criterion.eval()

    def scheduler_step(self, epoch_idx=None):
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(self._loss.average)
        else:
            self.scheduler.step()

    def forward(self, batch, batch_idx=None):
        x, y = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        return y_pred, loss

    def forward_pass_on_training_step(self, batch, batch_idx=None):
        if not self.model.training:
            self.log_warn("Training without self.train() call")
        y_pred, loss = self.forward(batch, batch_idx=batch_idx)
        self._push_for_backward(loss)
        self._loss.update(loss.item())
        return y_pred

    @torch.no_grad()
    def forward_pass_on_validation_step(self, batch, batch_idx=None):
        if self.model.training:
            self.log_warn("Evaluating without self.eval() call")
        y_pred, loss = self.forward(batch, batch_idx=batch_idx)
        self._loss.update(loss.item())
        return y_pred
    
    @torch.no_grad()
    def forward_pass_on_evauluation_step(self, batch, batch_idx=None):
        if self.model.training:
            self.log_warn("Evaluating without self.eval() call")
        y_pred, loss = self.forward(batch, batch_idx=batch_idx)
        self._loss.update(loss.item())
        return y_pred

    def _push_for_backward(self, tensor: Tensor):
        try:
            self._backward_hooks.append(
                GradTensor.field_validator(tensor)
            )
        except ValueError as e:
            self.log_debug(e)

    def backward_pass(self):
        self.optimizer.zero_grad()
        if self._backward_hooks.__len__() == 0:
            self.log_warn(
                "TrainerModel.backward_pass receives no loss to backward"
                "check requires_grad attribute of input and output pairs"
            )
        while self._backward_hooks.__len__() != 0:
            self._backward_hooks.pop().backward()
        self.optimizer.step()

    def reset_backward(self):
        if self._backward_hooks.__len__() != 0:
            self.log_warn("BackwardHook is not empty")
            self._backward_hooks.clear()
