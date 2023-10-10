import inspect
import logging
import pydantic
import typing
import warnings
import torch
from collections import OrderedDict

from torchutils.metrics import AverageScore
from .hashs import digest
from .typing import (
    NeuralNet,
    Criterion,
    Optimizer,
    Scheduler,
    Tensor,
    GradTensor,
)


class TrainerModel(pydantic.BaseModel):
    arguments: typing.Dict = pydantic.Field(default_factory=dict)
    criterion: Criterion
    model: NeuralNet
    optimizer: Optimizer
    scheduler: typing.Optional[Scheduler]
    _loss: AverageScore = pydantic.PrivateAttr()
    _logger: logging.Logger = pydantic.PrivateAttr()
    _backward_hooks: typing.List[GradTensor] = pydantic.PrivateAttr(default_factory=list)

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
        model: NeuralNet.TYPE,
        criterion: Criterion.TYPE,
        optimizer: Optimizer.TYPE,
        scheduler: typing.Optional[Scheduler.TYPE] = None,
        ** kwargs,
    ):
        super().__init__(model=model, arguments=kwargs,
                         criterion=criterion, optimizer=optimizer,
                         scheduler=scheduler)
        self._logger = logging.getLogger(model.__class__.__qualname__ + " Model")
        self._loss = AverageScore(model.__class__.__name__ + " Model Loss")


    def __getstate__(self) -> typing.Set[str]:
        if isinstance(self.criterion, torch.nn.Module):
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

    def __hash__(self) -> int:
        return int(digest(self.state_dict()), 16)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def checksum(self) -> str:
        return digest(self.state_dict())

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

    def train(self) -> None:
        self.model.train()
        if isinstance(self.criterion, torch.nn.Module):
            self.criterion.train()

    def eval(self) -> None:
        self.model.eval()
        if isinstance(self.criterion, torch.nn.Module):
            self.criterion.eval()

    def summary(self, input_size, batch_size=-1) -> None:
        try:
            import torchsummary
            torchsummary.summary(
                self.model, input_size,
                batch_size=batch_size,
                device=self.device.type)
        except ImportError:
            warnings.warn("torchsummary not installed", RuntimeWarning)
            print("torchsummary not installed")


    def scheduler_step(self, epoch_idx=None):
        if self.scheduler is None:
            pass
        elif isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(self._loss.average)
        else:
            self.scheduler.step()
        self._loss.reset()

    def reset_backward(self):
        if self._backward_hooks.__len__() != 0:
            warnings.warn("BackwardHook is not empty", RuntimeWarning)
            self._backward_hooks.clear()
        self._loss.reset()

    def _push_for_backward(self, tensor: GradTensor.TYPE) -> None:
        if GradTensor.isinstance(tensor):
            self._backward_hooks.append(tensor)
        else:
            warnings.warn("Tensor is not a GradTensor", RuntimeWarning)

    def forward_pass(self, batch, batch_idx=None):
        x, y = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        self._push_for_backward(loss)
        self._loss.update(loss.item())
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
