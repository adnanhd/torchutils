import pydantic
import time
import os
import warnings
from .pydantic_types import ModuleType, LossType, OptimizerType, SchedulerType
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Dict, Union
import torch.optim as optimizers


class TrainerArguments(pydantic.BaseModel):
    num_epochs: int = pydantic.Field(ge=0, description="Number of epochs")
    learning_rate: float = pydantic.Field(ge=0.0, le=1.0)
    train_dl_batch_size: int
    valid_dl_batch_size: int = -1 # All elements at a batch
    test_dl_batch_size: int = 1 # One element per patch
    num_epochs_per_validation: int
    class Config:
        allow_mutation = False


class TrainerModel(pydantic.BaseModel):
    model: ModuleType
    criterion: LossType
    optimizer: OptimizerType #Union[OptimizerType, str]
    scheduler: Optional[SchedulerType]

    @property
    def device(self):
        device = None
        for attr_name in self._get_attr_names():
            attr = getattr(self, attr_name)
            if attr is not None:
                if device is None:
                    device = attr.device
                elif attr.device != device:
                warnings.warn(f"{attr_name} is on different device.", RuntimeWarning)
        return device

    @device.setter
    def set_device(self, device):
        for attr_name in self._get_attr_names():
            attr = getattr(self, attr_name)
            if attr is not None:
                setattr(self, attr_name, attr.to(device=device))

    def configure_optimizers(self, lr: float, **kwargs):
        return optimizers.Adam(self.model.parameters(), lr=lr, **kwargs)

    def _get_attr_names(self):
        return ('model', 'optimizer', 'scheduler', 'criterion')

    def save_into_checkpoint(self, path, **state):
        dirname, basename = os.path.split(path)
        if dirname != '': os.makedirs(path, exist_ok=True)

        for key in self._get_attr_names():
            module = self.__getattribute__(key)
            if module is not None:
                if hasattr(module, 'state_dict'):
                    state[key] = module.state_dict()
                else:
                    warnings.warn(f"{key} has no state_dict() attribute.", RuntimeWarning)
        # TODO: add version stamping before here
        torch.save(state, path)

    def load_from_checkpoint(self, path: str) -> Dict[str, "torch.Tensor"]:
        checkpoint = torch.load(path, map_location=self.device)
        for key in self._get_attr_names():
            if module is not None and key in checkpoint:
                if hasattr(module, 'load_state_dict'):
                    self.__getattribute__(key).load_state_dict(checkpoint.pop(key))
                else:
                    warnings.warn(f"{key} exits in the checkpoint but that of {self.__qualname__} \
                            has no load_state_dict() attribute.", RuntimeWarning)
        return checkpoint

    def training_step(self, batch, batch_idx: int):
        ...

    def validating_step(self, batch, batch_idx: int):
        ...

    def evaluating_step(self, batch, batch_idx: int):
        ...


class TrainerCallbackArguments(pydantic.BaseModel):
    train_dl_length: int
    valid_dl_length: int
    test_dl_length: int
    args: TrainerArguments
    model: TrainerModel

