import pydantic
import time
import torch
import os
import warnings
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Dict, Union, Any, List
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
    optimizer: Optional[OptimizerType] #Union[OptimizerType, str]
    scheduler: Optional[SchedulerType]
    device: Any = None

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype
    
    @property
    def device(self):
        return next(self.model.parameters()).device

    def __setattr__(self, key, value):
        if key == 'device':
            self.model = self.model.to(device=value)            
            return object.__setattr__(self.model, 'device', value)
        if key == 'dtype':
            self.model = self.model.to(dtype=value)            
            return object.__setattr__(self.model, 'dtype', value)
        return super().__setattr__(key, value)

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
        self.optimizer = optimizers.Adam(self.model.parameters(), lr=lr, **kwargs)

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

    def detached_step(self, x, y, batch_idx):
        y_pred = self.model_fn(batch=x, batch_idx=batch_idx)
        loss = self.loss_fn(batch=y_pred, batch_true=y, batch_idx=batch_idx)
        return y_pred, loss


class TrainingArguments(pydantic.BaseModel):
    class Config:
        allow_mutation = False

    num_epochs: int = pydantic.Field(ge=0, description="Number of epochs")
    learning_rate: float = pydantic.Field(ge=0.0, le=1.0)
    resume_epochs: int = 0
    train_dl_batch_size: int
    valid_dl_batch_size: int = -1 # All elements at a batch
    num_epochs_per_validation: int = 1

class EvaluatingArguments(pydantic.BaseModel):
    class Config:
        allow_mutation = False
    eval_dl_batch_size: int = 1 # One element per patch



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
        raise LookupError(f"HandlerArguments has been locked already.")


class StepResults(pydantic.BaseModel):
    x: NpTorchType
    y_true: NpTorchType
    y_pred: NpTorchType
    class Config:
        allow_mutation = False


class EpochResults(pydantic.BaseModel):
    pass
    #x: NpTorchType
    #y_true: NpTorchType
    #y_pred: NpTorchType
    #class Config:
    #    allow_mutation = False


class HandlerArguments(pydantic.BaseModel):
    args: Union[TrainingArguments, EvaluatingArguments] = None
    model: TrainerModel
    status_ptr: List[TrainerStatus]
    train_dl: Optional[TrainerDataLoader] = None
    valid_dl: Optional[TrainerDataLoader] = None
    eval_dl: Optional[TrainerDataLoader] = None

    @property
    def status(self):
        return self.status_ptr[0]

    def update_status(self, batch=None, epoch=None):
        if batch is not None: self.status_ptr[0].current_batch = batch
        if epoch is not None: self.status_ptr[0].current_epoch = epoch

    def reset_status(self):
        self.status_ptr[0].current_epoch = None
        self.status_ptr[0].current_batch = None

    class Config:
        allow_mutation = True

    def set_arguments(self):
        if self.__config__.allow_mutation:
            self.__config__.allow_mutation = False
            def arguments_setter_callback(
                    new_args: Union[TrainingArguments, EvaluatingArguments],
                    dataloaders: Dict[str, Optional[TrainerDataLoader]] = dict()):
                self.__config__.allow_mutation = True
                self.args = new_args
                for dl_name in ('train_dl', 'valid_dl', 'eval_dl'):
                    dl = dataloaders.setdefault(dl_name, None)
                    if dl is not None:
                        dl = TrainerDataLoader(dataloader=dl)
                        self.__setattr__(dl_name, dl)
                self.__config__.allow_mutation = False
            return arguments_setter_callback
        raise LookupError(f"HandlerArguments has been locked already.")

    @pydantic.validator('train_dl', 'valid_dl', 'eval_dl')
    def validate_dataloaders(cls, field_type):
        if isinstance(field_type, TrainerDataLoader):
            return field_type
        elif isinstance(field_type, torch.utils.data.DataLoader):
            return TrainerDataLoader(dataloader=field_type)
        else:
            raise ValueError(f'Not possible to accept a type of {type(field_type)}')

    @property
    def is_train(self):
        if isinstance(self.args, TrainerDataLoader):
            return True
        elif isinstance(self.args, EvaluatingArguments):
            return False
        else:
            return None



