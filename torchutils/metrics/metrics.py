from lib2to3.pgen2.token import OP
from optparse import Option
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Mapping, Optional, Union, Callable, Tuple, Iterable
from .handler.events import on_initialization, on_run_begin, on_run_end, on_step_begin, on_step_end


class TrainerMetric(ABC):

    def __init__(self,
                 metrics: Mapping[str, Callable[[
                     torch.Tensor, torch.Tensor], float]] = dict(),
                 loss_list=None,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None
                 ):
        self.metrics = metrics
        self.index = set()
        self.loss_list = loss_list
        self.device = device
        self.dtype = dtype

    # @on_run_begin
    def init(self,
             step_size: int, # number of batches in an epoch
             batch_size: int = None, # number of samples in a batch
             epoch: int = None,
             **kwargs,
             ) -> None:
        if self.dtype is not None:
            kwargs.setdefault('dtype', self.dtype)
        if self.device is not None:
            kwargs.setdefault('device', self.device)

        self.loss_list = torch.zeros(
            step_size, len(self.metrics), **kwargs,
        )

    # @on_epoch_begin
    def reset(self):
        self.loss_list.zero_()
        self.index.clear()

    # @on_epoch_end
    def update(self,
               index: Optional[int] = None,
               logger=None,
               *args, **kwargs):
        if logger is not None:
            logger.log(**self.updated_values())

    # @on_step_end
    def step(
        self,
        batch_idx: int,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        *args,
        **kwargs,
    ) -> None:
        for metric_idx, metric_fn in enumerate(self.metrics.values()):
            self.loss_list[batch_idx, metric_idx] = metric_fn(y_true, y_pred)

    def stepped_values(self, batch_idx: int) -> Mapping[str, torch.tensor]:
        if batch_idx not in self.index:
            return {}
        return dict(zip(self.metrics.keys(), self.loss_list[batch_idx, :]))

    def updated_values(self, *args, **kwargs) -> Mapping[str, torch.Tensor]:
        if len(self.index) == 0:
            return {}
        return dict(zip(self.metrics.keys(), self.loss_list[list(self.index)].mean(0)))

    def save(self):
        with open(self.path, "w") as f:
            f.write("\n".join(self.logs))


def HookerClass(cls):
    def hooked_class(*args, **kwargs):
        obj = cls(*args, **kwargs)
        on_initialization(obj.init)
        on_run_begin(obj.reset)
        on_run_end(obj.update)
        # on_step_begin()
        on_step_end(obj.step)
        return obj
    return hooked_class
