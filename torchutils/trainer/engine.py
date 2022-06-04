#!/usr/bin/env python3
import os
from torchutils.data.dataset import Dataset
from torchutils.metrics import TrainerMetric
import torch
import numpy as np
from torchutils.callbacks import (
    CallbackHandler,
    CallbackMethodNotImplementedError,
    TrainerCallback,
    StopTrainingError,
)

import warnings
#from torchutils.metrics import loss_to_metric, MetricHandler
from .handler import TrainerHandler
from torchutils.utils import Version
from torchutils.utils.pydantic import (
        TrainerModel, 
        TrainingArguments, 
        EvaluatingArguments,
        TrainerStatus
)

from typing import (
    List, 
    Dict, 
    Any, 
    Mapping, 
    Optional, 
    Union, 
    Callable, 
    Tuple, 
    Iterable,
    Set
)

version = Version('1.2.0')

class Trainer:
    __slots__ = ['ytype', '_model', '_handler', 'status']
    def __init__(
        self,
        model: Union[TrainerModel, torch.nn.Module],
        loss: Optional[Union[torch.autograd.Function, torch.nn.Module]] = None,
        device=None,  # todo: remove
        xtype: Union[torch.dtype, np.dtype, type] = torch.float32,
        ytype: Union[torch.dtype, np.dtype, type] = torch.float32,
        callbacks: List[TrainerCallback] = list(),
        metrics: Union[Set[str], List[str], Tuple[str]] = set(),
        **kwargs,
    ):
        assert isinstance(model, TrainerModel) or loss is not None
        if not isinstance(model, TrainerModel):
            model = TrainerModel(model=model, criterion=loss)

        self._model = model
        self._model.device = device

        self._model.dtype = xtype
        self.ytype = ytype

        self.status = TrainerStatus()
        self._handler = TrainerHandler(model=self._model, status_ptr=[self.status])

    @property
    def xtype(self):
        return self._model.dtype

    @property
    def device(self) -> torch.dtype:
        return self._model.device

    def create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        train_mode: bool = True,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        assert isinstance(dataset, torch.utils.data.Dataset), """type(dataset), \
                {} must be inherited from torch.utils.data.Dataset""".format(type(dataset))

        try:
            if isinstance(dataset.features, np.ndarray) and self.device != torch.device('cpu'):
                dataset.features = torch.from_numpy(dataset.features)
            dataset.features = dataset.features.to(device=self.device, dtype=self.xtype)
            if isinstance(dataset.labels, np.ndarray) and self.device != torch.device('cpu'):
                dataset.labels = torch.from_numpy(dataset.labels)
            dataset.labels = dataset.labels.to(device=self.device, dtype=self.ytype)
        except AttributeError:
            warnings.warn("Using a Dataset not derived from torchutils.data.Dataset is dangerous for dtype integrity")
        
        kwargs.setdefault('shuffle', train_mode)
        kwargs.setdefault('pin_memory', not torch.cuda.is_available())
        kwargs.setdefault('num_workers', 0 if torch.cuda.is_available() else os.cpu_count())
        if train_mode == False or batch_size is None:
            kwargs['batch_size'] = dataset.__len__()
        else:
            kwargs['batch_size'] = batch_size
        
        if isinstance(dataset, Dataset):
            return dataset.dataloader(**kwargs)
        else:
            return torch.utils.data.DataLoader(dataset, **kwargs)

    def compile(
        self,
        device=None,
        model=None,
        loss=None,
        optim=None,
        sched=None,
        metrics=list(),
        loggers=list(),
        callbacks=list(),
        **parameters # trainer parameters
    ) -> None:
        if device is not None: self._model.device = device
        self._model.compile(model, optim, loss, sched)
        self._handler.compile(loggers, metrics, callbacks)

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        train_dataset: torch.utils.data.Dataset,
        learning_rate: float = None,
        valid_dataset: Optional[torch.utils.data.Dataset] = None,
        train_dataloader_kwargs: Optional[Mapping] = {},
        valid_dataloader_kwargs: Optional[Mapping] = {},
        **kwargs
    ):
        if learning_rate is not None:
            self._model.configure_optimizers(lr=learning_rate)

        train_dl = self.create_dataloader(
            dataset=train_dataset, train_mode=True,
            batch_size=batch_size, **train_dataloader_kwargs,
        )
        
        valid_dl = None
        if valid_dataset is not None:
            valid_dl = self.create_dataloader(dataset=valid_dataset, 
                    train_mode=False, **valid_dataloader_kwargs,
                batch_size=kwargs.setdefault('valid_dl_batch_size', -1))
            kwargs['valid_dl_batch_size'] = valid_dl.batch_size

        args = TrainingArguments(num_epochs=num_epochs, 
                learning_rate=learning_rate, **kwargs,
                train_dl_batch_size=train_dl.batch_size)

        self._handler.set_arguments(args, 
                train_dl=train_dl, valid_dl=valid_dl)
        self._handler.on_initialization()

        try:
            return self._run_training(args, train_dl, valid_dl)
        except StopTrainingError:
            self._handler.on_stop_training_error()
        finally:
            self._handler.on_termination()
        
    def evaluate(
        self,
        dataset,
        dataloader_kwargs={},
        **kwargs
    ):
        eval_dl = self.create_dataloader(
            dataset=dataset,
            train_mode=False,
            batch_size=dataset.__len__(),
            **dataloader_kwargs,
        )
        
        args = EvaluatingArguments(**kwargs,
                eval_dl_batch_size=eval_dl.batch_size)
        
        self._handler.set_arguments(args, eval_dl=eval_dl)
        self._handler.on_initialization()

        try:
            return self._run_evaluating(eval_dl)
        except StopTrainingError:
            self._handler.on_stop_training_error()
        finally:
            self._handler.on_termination()

    from .train import _run_training
    from .valid import _run_validating
    from .eval import _run_evaluating    

