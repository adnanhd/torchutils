import logging
import inspect
import pydantic
import typing
import torch
from collections import OrderedDict

from torchutils.metrics import TrainerAverageScore
from .hashs import digest
from .typing import (
    NeuralNet,
    Criterion,
    Functional,
    Optimizer,
    Scheduler,
    Tensor,
    GradTensor,
)


class TrainerModel(pydantic.BaseModel):
    arguments: typing.Dict = pydantic.Field(default_factory=dict)
    criterion: typing.Union[Criterion, Functional]
    model: NeuralNet
    optimizer: Optimizer
    scheduler: typing.Optional[Scheduler]
    _scores: typing.Dict[str, TrainerAverageScore] = pydantic.PrivateAttr(default_factory=dict)
    _buffer: typing.Dict[str, typing.Any] = pydantic.PrivateAttr(default_factory=dict)
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
        model: NeuralNet,
        criterion: Criterion,
        optimizer: Optimizer,
        modelname: str = None,
        scheduler: typing.Optional[Scheduler] = None,
        ** kwargs,
    ):
        super().__init__(model=model, arguments=kwargs,
                         criterion=criterion, optimizer=optimizer,
                         scheduler=scheduler)
        if modelname is None:
            modelname = model.__class__.__qualname__
        loggername = self.__class__.__module__ + '.' + self.__class__.__name__
        if inspect.isfunction(self.criterion):
            lossname = self.criterion.__name__
        else:
            lossname = self.criterion.__class__.__name__
        self._scores['loss'] = TrainerAverageScore(lossname, reset_on='epoch_end')
        self._logger = logging.getLogger(loggername)

    # score functions
    def register_score(self, name: str, score: TrainerAverageScore):
        assert isinstance(score, TrainerAverageScore)
        assert name not in self.__pydantic_private__['_scores'].keys()
        return self.__pydantic_private__['_scores'].__setitem__(name, score)

    def get_score(self, name: str):
        assert name in self.__pydantic_private__['_scores'].keys()
        return self.__pydantic_private__['_scores'].__getitem__(name)

    def get_score_names(self) -> typing.Set[str]:
        return set(s.name for s in self._scores.values())

    # handler functions
    def add_handlers(self, handlers=list()):
        for hdlr in handlers:
            self.__pydantic_private__['_logger'].addHandler(hdlr)

    # @TODO: remove_handlers -> pop_handlers
    def remove_handlers(self, handlers=list()):
        for hdlr in handlers:
            self.__pydantic_private__['_logger'].removeHandler(hdlr)

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
                self._logger.warn(
                    f"{key} has no state_dict() attribute."
                )
        return state

    def load_state_dict(
            self,
            state_dict: typing.Dict[str, "torch.Tensor"]) -> None:
        for key in self.__state_names__():
            module = getattr(self, key)
            if module is None or key not in state_dict:
                self._logger.warn(
                    f"{key} either absent in the state_dict or None."
                )
            elif hasattr(module, 'load_state_dict'):
                module.load_state_dict(state_dict[key])
            else:
                self._logger.warn(
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

    @property
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
            self._logger.warn("torchsummary not installed")
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
            self.scheduler.step(self._scores['loss'].average)
        else:
            self.scheduler.step()

    def forward(self, batch, batch_idx=None):
        x, y = batch
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        return y_pred, loss

    def forward_pass(self, batch, batch_idx=None):
        if not self.model.__getstate__()['training']:
            self._logger.warn("Training without self.train() call")
        y_pred, loss = self.forward(batch, batch_idx=batch_idx)
        self._push_for_backward(loss)
        self._scores['loss'].update(loss.item())
        return y_pred

    @torch.no_grad()
    def forward_pass_no_grad(self, batch, batch_idx=None):
        if self.model.__getstate__()['training']:
            self._logger.warn("Evaluating without self.eval() call")
        y_pred, loss = self.forward(batch, batch_idx=batch_idx)
        self._scores['loss'].update(loss.item())
        return y_pred

    def _push_for_backward(self, tensor: Tensor):
        try:
            tensor = GradTensor.tensor_validator(tensor)
            tensor = GradTensor.grad_validator(tensor)
            self._backward_hooks.append(tensor)
        except ValueError as e:
            self._logger.debug(e)

    def backward_pass(self):
        self.optimizer.zero_grad()
        if self._backward_hooks.__len__() == 0:
            self._logger.warn(
                "TrainerModel.backward_pass receives no loss to backward"
                "check requires_grad attribute of input and output pairs"
            )
        while self._backward_hooks.__len__() != 0:
            self._backward_hooks.pop().backward()
        self.optimizer.step()

    def reset_backward(self):
        if self._backward_hooks.__len__() != 0:
            self._logger.warn("BackwardHook is not empty")
            self._backward_hooks.clear()
