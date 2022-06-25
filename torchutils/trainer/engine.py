#!/usr/bin/env python3
import os
from torchutils.data.dataset import Dataset
import torch
import numpy as np
from torchutils.callbacks import (
    TrainerCallback,
    StopTrainingError,
)

import warnings
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
    Iterable,
    Mapping,
    Optional,
    Union,
    Tuple,
    Set
)

version = Version('1.2.0')


class Trainer:
    __slots__ = ['ytype', '_model', '_handler']

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

        self._model: TrainerModel = model
        self._model.device = device

        self._model.dtype = xtype
        self.ytype = ytype

        self._handler = TrainerHandler()
        if model is not None:
            self._model.register_metrics_to(self._handler._metrics)

    @property
    def status(self) -> TrainerStatus:
        return self._handler.status

    @property
    def hyperparams(self) -> Union[TrainingArguments, EvaluatingArguments]:
        return self._handler.hparams

    @property
    def xtype(self) -> torch.dtype:
        return self._model.dtype

    @property
    def device(self) -> torch.device:
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
            dataset.features = dataset.features.to(
                device=self.device, dtype=self.xtype)
            if isinstance(dataset.labels, np.ndarray) and self.device != torch.device('cpu'):
                dataset.labels = torch.from_numpy(dataset.labels)
            dataset.labels = dataset.labels.to(
                device=self.device, dtype=self.ytype)
        except AttributeError:
            warnings.warn(
                "Using a Dataset not derived from torchutils.data.Dataset is dangerous for dtype integrity")

        kwargs.setdefault('shuffle', train_mode)
        kwargs.setdefault('pin_memory', not torch.cuda.is_available())
        kwargs.setdefault(
            'num_workers', 0 if torch.cuda.is_available() else os.cpu_count())
        if not train_mode or batch_size is None:
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
        loggers=dict(),
        callbacks=list(),
        **parameters  # trainer parameters
    ) -> None:
        if device is not None:
            self._model.device = device
        if model is not None:
            self._model.register_metrics_to(self._handler._metrics)
        self._model.compile(model, loss, optim, sched)
        self._handler.compile_handlers(loggers, metrics, callbacks)

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        train_dataset: torch.utils.data.Dataset,
        learning_rate: float = None,
        valid_dataset: Optional[torch.utils.data.Dataset] = None,
        train_dataloader_kwargs: Optional[Mapping] = {},
        valid_dataloader_kwargs: Optional[Mapping] = {},
        **hparams
    ):
        if learning_rate is not None:
            self._model.learning_rate = learning_rate
        else:
            learning_rate = self._model.learning_rate

        train_dl = self.create_dataloader(
            dataset=train_dataset, train_mode=True,
            batch_size=batch_size, **train_dataloader_kwargs,
        )

        valid_dl = None
        if valid_dataset is not None:
            valid_dl = self.create_dataloader(dataset=valid_dataset,
                                              train_mode=False,
                                              **valid_dataloader_kwargs,
                                              batch_size=hparams.setdefault(
                                                  'valid_dl_batch_size', -1)
                                              )
            hparams['valid_dl_batch_size'] = valid_dl.batch_size

        self._handler.compile_model_and_hparams(
            model=self._model,
            train_dl=train_dl,
            valid_dl=valid_dl,
            num_epochs=num_epochs,
            **hparams
        )

        try:
            self._handler._arguments.set_status(
                epoch=self.hyperparams.resume_epochs)
            self._handler.on_initialization()
            results = self._run_training(
                train_loader=train_dl,
                valid_loader=valid_dl
            )
        except StopTrainingError:
            self._handler.on_stop_training_error()
        else:
            return results
        finally:
            self._handler._arguments.reset_status()
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

        self._handler.compile_model_and_hparams(
            model=self._model,
            eval_dl=eval_dl,
            eval_dl_batch_size=eval_dl.batch_size,
            **kwargs
        )

        try:
            self._handler.on_initialization()
            results = self._run_evaluating(eval_dl)
        except StopTrainingError:
            self._handler.on_stop_training_error()
        else:
            return results
        finally:
            self._handler.on_termination()
            self._handler._arguments.reset_status()

    def _run_training(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        self._handler.on_training_begin()

        for epoch in range(self.hyperparams.resume_epochs,
                           self.hyperparams.num_epochs):
            self._handler._arguments.set_status(epoch=epoch)
            self._run_training_epoch(train_loader, valid_loader)

        self._handler.on_training_end()
        return self._handler.trainer_proxy.get_score_history()

    def _run_training_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> torch.Tensor:
        self._handler.on_training_epoch_begin()
        self._model.train()

        for batch_idx, batch in enumerate(train_loader):
            self._run_training_step(batch, batch_idx)

        self._model.reset_backward()
        if valid_loader is not None and (self.status.current_epoch + 1) % \
                self.hyperparams.num_epochs_per_validation == 0:
            self._run_validating(valid_loader)

        self._handler.on_training_epoch_end()

    def _run_training_step(
        self,
        batch: Iterable[torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        x = batch[0].to(device=self.device, dtype=self.xtype)
        y = batch[1].to(device=self.device, dtype=self.ytype)

        self._handler._arguments.set_status(batch=batch_idx)
        self._handler.on_training_step_begin()

        y_pred = self._model.forward_pass(x, y, self.status.current_batch)
        self._model.backward_pass()

        self._handler.on_training_step_end(x=x, y=y, y_pred=y_pred)

        return y_pred.detach()

    @torch.no_grad()
    def _run_evaluating(
        self,
        eval_loader: Optional[torch.utils.data.DataLoader],
    ) -> torch.Tensor:

        self._handler.on_evaluation_run_begin()
        self._model.eval()

        preds = []
        for batch_idx, batch in enumerate(eval_loader):
            pred = self._run_evaluating_step(batch_idx, batch)
            preds.extend(pred)

        self._model.reset_backward()
        self._handler.on_evaluation_run_end()
        return preds

    @torch.no_grad()
    def _run_evaluating_step(
        self,
        batch_idx: int,
        batch: Iterable[torch.Tensor]
    ) -> torch.Tensor:
        x = batch[0].to(device=self.device, dtype=self.xtype)
        y = batch[1].to(device=self.device, dtype=self.ytype)

        self._handler._arguments.set_status(batch=batch_idx)
        self._handler.on_evaluation_step_begin()

        y_pred = self._model.forward_pass(
            x=x, y=y, batch_idx=self.status.current_batch
        )

        self._handler.on_evaluation_step_end(x=x, y=y, y_pred=y_pred)

        return y_pred.detach()

    @torch.no_grad()
    def _run_validating(
        self,
        valid_loader: Optional[torch.utils.data.DataLoader],
    ) -> torch.Tensor:

        self._handler.on_validation_run_begin()
        self._model.eval()

        for step_idx, batch in enumerate(valid_loader):
            self._run_validating_step(step_idx, batch)

        self._model.reset_backward()
        self._handler.on_validation_run_end()

    @torch.no_grad()
    def _run_validating_step(
        self,
        batch_idx: int,
        batch: Iterable[torch.Tensor]
    ) -> torch.Tensor:
        x = batch[0].to(device=self.device, dtype=self.xtype)
        y = batch[1].to(device=self.device, dtype=self.ytype)

        self._handler._arguments.set_status(batch=batch_idx)
        self._handler.on_validation_step_begin()

        y_pred = self._model.forward_pass(
            x=x, y=y, batch_idx=self.status.current_batch
        )

        self._handler.on_validation_step_end(x=x, y=y, y_pred=y_pred)

        return y_pred.detach()
