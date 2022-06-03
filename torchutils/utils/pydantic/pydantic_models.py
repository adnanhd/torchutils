import pydantic
import time
import torch
import os
import warnings
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Dict, Union, Any
import torch.optim as optimizers
from .pydantic_types import (
        NpScalarType,
        NpTorchType,
        ModuleType, 
        LossType, 
        OptimizerType, 
        SchedulerType, 
        DataLoaderType, 
        DatasetType
)
import torchsummary

class TrainerModel(pydantic.BaseModel):
    model: ModuleType
    criterion: LossType
    optimizer: OptimizerType #Union[OptimizerType, str]
    scheduler: Optional[SchedulerType]
    device: Any = None

    #@property
    #def device(self):
    #    return next(self.model.parameters()).device

    @pydantic.validator('device')
    def set_device(cls, field_type):
        if isinstance(field_type, torch.device):
            self.model = self.model.to(device=device)
            return field_type
        raise ValueError()

    @classmethod
    def _get_attr_names(cls):
        return ('model', 'optimizer', 'scheduler', 'criterion')

    def save_into_checkpoint(self, path, **state):
        if os.path.split(path)[0] != '': 
            os.makedirs(path, exist_ok=True)
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
            module = getattr(self, key)
            if module is not None and key in checkpoint:
                if hasattr(module, 'load_state_dict'):
                    module.load_state_dict(checkpoint.pop(key))
                else:
                    warnings.warn(f"{key} exits in the checkpoint but that \
                            of {self.__qualname__} has no load_state_dict() \
                            attribute.", RuntimeWarning)
        return checkpoint

    def summary(self, input_size, batch_size=-1):
        torchsummary.summary(
                self.model, input_size, 
                batch_size=batch_size, 
                device=self.device)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def configure_optimizers(self, lr: float, **kwargs):
        return optimizers.Adam(self.model.parameters(), lr=lr, **kwargs)

    def compile(self, 
            model: Module=None, optim: Union[str, Optimizer]=None,
            loss: Module=None,  sched: Union[str, _LRScheduler]=None):
        if model is not None: self.model = model
        if loss is not None:  self.criterion = loss
        if optim is not None: self.optimizer = optim
        if sched is not None: self.scheduler = sched

    def optimizer_step(self):
        if self.optimizer is not None: self.optimizer.step()

    def optimizer_zero_grad(self):
        if self.optimizer is not None: self.optimizer.zero_grad()

    def scheduler_step(self):
        if self.scheduler is not None: self.scheduler.step()

    def model_fn(self, batch, batch_idx: int):
        return self.model(batch)

    def validating_step(self, batch, batch_idx: int):
        return self.model(batch.detach()).detach()

    def loss_fn(self, batch, batch_true, batch_idx: int):
        return self.criterion(input=batch, target=batch_true)

    def attached_step(self, x, y, batch_idx):
        self.optimizer_zero_grad()
        y_pred = self.model_fn(batch=x, batch_idx=batch_idx)
        loss = self.loss_fn(batch=y_pred, batch_true=y, batch_idx=batch_idx)
        loss.backward()
        self.optimizer_step()
        return y_pred, loss

    def detached_step(self, x, y, idx):
        y_pred = self.model_fn(batch=x, batch_idx=idx)
        loss = self.loss_fn(batch=y_pred, batch_true=y)
        return y_pred, loss


class TrainerArguments(pydantic.BaseModel):
    class Config:
        allow_mutation = False

    num_epochs: int = pydantic.Field(ge=0, description="Number of epochs")
    learning_rate: float = pydantic.Field(ge=0.0, le=1.0)
    resume_epochs: int = 0
    train_dl_batch_size: int
    valid_dl_batch_size: int = -1 # All elements at a batch
    test_dl_batch_size: int = 1 # One element per patch
    num_epochs_per_validation: int = 1


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
        allow_mutation = False
    current_epoch: int = None
    current_batch: int = None
    
    def lock(self):
        if self.__config__.allow_mutation:
            self.__config__.allow_mutation = False
            def unlock(self):
                self.__config__.allow_mutation = True
            return unlock
        raise LookupError(f"HandlerArguments has been locked already.")

    def step(self, n=1):
        self.current_batch += n

    def epoch(self):
        self.current_epoch += 1

    def update(self, batch=None, epoch=None):
        if batch is not None: self.current_batch = batch
        if epoch is not None: self.current_epoch = epoch

    def reset(self):
        self.current_epoch = None
        self.current_batch = None


class StepResults(pydantic.BaseModel):
    x: NpTorchType
    y_true: NpTorchType
    y_pred: NpTorchType
    class Config:
        allow_mutation = False

    #@pydantic.validator('x', 'y_true', 'y_pred')
    #def validate_all(cls, field_type):
    #    for validator in NpTorchType.__get_validators__():
    #        if validator(field_type):
    #            pass


class EpochResults(pydantic.BaseModel):
    pass
    #x: NpTorchType
    #y_true: NpTorchType
    #y_pred: NpTorchType
    #class Config:
    #    allow_mutation = False


class HandlerArguments(pydantic.BaseModel):
    args: TrainerArguments
    model: TrainerModel
    status: TrainerStatus
    train_dl: Optional[TrainerDataLoader] = None
    valid_dl: Optional[TrainerDataLoader] = None
    test_dl: Optional[TrainerDataLoader] = None

    class Config:
        allow_mutation = True

    def lock(self):
        if self.__config__.allow_mutation:
            self.__config__.allow_mutation = False
            def unlock(self):
                self.__config__.allow_mutation = True
            return unlock
        raise LookupError(f"HandlerArguments has been locked already.")

    @pydantic.validator('train_dl', 'valid_dl', 'test_dl')
    def validate_dataloaders(cls, field_type, **kwargs):
        if isinstance(field_type, TrainerDataLoader):
            return field_type
        elif isinstance(field_type, torch.utils.data.DataLoader):
            return TrainerDataLoader(dataloader=field_type)
        else:
            raise ValueError(f'Not possible to accept a type of {type(field_type)}')



