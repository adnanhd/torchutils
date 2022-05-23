#!/usr/bin/env python3
import os
from ..data.dataset import Dataset
from ..metrics import TrainerMetric
import torch
from torchutils.callbacks import (
    CallbackHandler,
    CallbackMethodNotImplementedError,
    TrainerCallback,
    StopTrainingError,
)

import warnings
#from torchutils.metrics import loss_to_metric, MetricHandler
from .train import _run_training
from .valid import _run_validating
from .eval import _run_evaluating

from typing import (
    List, 
    Dict, 
    Any, 
    Mapping, 
    Optional, 
    Union, 
    Callable, 
    Tuple, 
    Iterable
)

version = '2.1.0'
def makedirs(path, verbose=False):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,  
        loss: Union[Callable, torch.nn.modules.loss._Loss],
        sched: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        model_name: str = "model",
        model_path: Optional[str] = "./checkpoints/",
        device=None,  # todo: remove
        xtype: Union[torch.dtype, type] = torch.float32,
        ytype: Union[torch.dtype, type] = torch.float32,
        *args,
        **kwargs,
    ):
        self.model_path = model_path
        self.model_name = model_name

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device)

        self.xtype = xtype
        self.ytype = ytype

        self.model = model #.to(device=self.device, dtype=self.xtype)
        self.criterion = loss
        self.optimizer = optim
        self.scheduler = sched

        self.metrics = TrainerMetric()
        #self.loggers = LoggerHandler()
        self.callbacks = CallbackHandler()

    @property
    def model_device(self):
        return next(self.model.parameters()).device

    def _prepare(self, *args):
        prepared = []
        for arg in args:
            if isinstance(arg, torch.nn.Module):
                arg = arg.to(device=self.device, dtype=self.xtype)
            elif isinstance(arg, Dataset):
                arg.features = arg.features.to(device=self.device, dtype=self.xtype)
                arg.labels = arg.labels.to(device=self.device, dtype=self.ytype)
            prepared.append(arg)
        
        if len(prepared) == 1:
            return prepared[0]
        else:
            return tuple(prepared)

    # if loss is not then model saves the best results only
    def save_checkpoint(self, path=None, **state):
        makedirs(self.model_path)
        if path is None:
            path = self.model_path
        if os.path.isdir(path):
            path = os.path.join(path, f"{self.model_name}.ckpt")

        for key in ('model', 'optimizer', 'scheduler', 'criterion'):
            module = self.__getattribute__(key)
            try:
                if module is not None: 
                    state[key] = module.state_dict()
            except AttributeError:
                warnings.warn(f"{key} has no state_dict() attribute.", RuntimeWarning)

        state['version'] = version
        torch.save(state, path)

    def load_checkpoint(self, epoch=None, path=None):
        if path is None:
            path = self.model_path

        if os.path.isdir(path):
            path = os.path.join(path, f"{self.model_name}.ckpt")

        checkpoint = torch.load(path, map_location=self.device)
        checkkeys = ("model", "scheduler", "optimizer", "criterion")

        print(checkpoint.keys())
        for key in checkkeys:
            if key in checkpoint:
                self.__getattribute__(key).load_state_dict(checkpoint[key])
            else:
                warnings.warn(f"{key} has no key in the loaded path.", RuntimeWarning)

                # del checkpoint[key]

        return epoch  # , pd.DataFrame(checkpoint, columns=checkpoint.keys(), index=range(epoch))

    # TODO: rewrite staticmethod again
    #@staticmethod
    def create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        train_mode: bool = True,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        assert isinstance(dataset, torch.utils.data.Dataset), """type(dataset), {} must be a descendant of torch.utils.data.Dataset""".format(type(dataset))
        
        try:
            dataset.features = dataset.features.to(device=self.device, dtype=self.xtype)
            dataset.labels = dataset.labels.to(device=self.device, dtype=self.ytype)
        except AttributeError:
            warnings.warn("Not using a conventional dataset derived from utils.data.Dataset")

        kwargs.setdefault('shuffle', train_mode)
        kwargs.setdefault('pin_memory', not torch.cuda.is_available())
        kwargs.setdefault('num_workers', 0 if torch.cuda.is_available() else os.cpu_count())
        if (not train_mode) or (batch_size is None):
            batch_size = dataset.__len__()

        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kwargs)

    def compile(
        self,
        device=None,
        model=None,
        loss=None,
        optim=None,
        sched=None,
        metrics=None,
        loggers=None,
        callbacks=None,
    ) -> None:
        if metrics is not None:
            self.metrics = TrainerMetric(metrics=metrics, dtype=self.ytype, device=self.device)

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: Optional[torch.utils.data.Dataset] = None,
        train_dataloader_kwargs: Optional[Mapping] = {},
        valid_dataloader_kwargs: Optional[Mapping] = {},
        callbacks: List[TrainerCallback] = [],
    ):
        self.callbacks.add_callbacks(callbacks)
        self.model = self._prepare(self.model)

        _train_dataloader = self.create_dataloader(
            dataset=train_dataset,
            train_mode=True,
            batch_size=batch_size,
            **train_dataloader_kwargs,
        )
        
        if valid_dataset is not None:
            _valid_dataloader = self.create_dataloader(
                dataset=valid_dataset,
                train_mode=False,
                batch_size=batch_size,
                **valid_dataloader_kwargs,
            )
        else:
            _valid_dataloader = None

        _run_training(
            self, num_epochs=num_epochs,
            train_loader=_train_dataloader,
            valid_loader=_valid_dataloader,
        )

        self.callbacks.remove_callbacks(callbacks)
        
    def evaluate(
        self,
        dataset,
        dataloader_kwargs={},
        callbacks: List[TrainerCallback] = [],
        **kwargs,
    ):
        self.callbacks.add_callbacks(callbacks)
        self.model = self._prepare(self.model)
        eval_dataloader = self.create_dataloader(
            dataset=dataset,
            train_mode=False,
            batch_size=dataset.__len__(),
            **dataloader_kwargs,
        )

        evals = _run_evaluating(self, eval_dataloader, **kwargs)
        self.callbacks.remove_callbacks(callbacks)
        return evals, self.metrics.updated_values()

    def __handle__(self, event, **kwargs):
        self.callbacks.call_event(self, event, **kwargs)

