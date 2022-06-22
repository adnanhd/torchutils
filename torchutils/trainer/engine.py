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
from torchutils.metrics import AverageMeter
from torchutils.utils import Version
from torchutils.utils.pydantic import (
    TrainerModel,
    TrainingArguments,
    EvaluatingArguments,
    TrainerStatus
)

from typing import (
    List,
    Mapping,
    Optional,
    Union,
    Tuple,
    Set
)

version = Version('1.2.0')


class Trainer:
    __slots__ = ['ytype', '_model', '_handler', '_loss_tracker']

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
        self._loss_tracker = AverageMeter('Loss')
        self._handler._metrics.add_score_meters(self._loss_tracker)

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
            learning_rate=learning_rate,
            train_dl_batch_size=train_dl.batch_size,
            **hparams
        )
        self._handler.on_initialization()

        try:
            return self._run_training(
                train_loader=train_dl,
                valid_loader=valid_dl
            )
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

    def _run_training(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        self._handler.on_training_begin()
        self._handler._arguments.set_status(
            epoch=self.hyperparams.resume_epochs)

        for epoch in range(self.hyperparams.resume_epochs,
                           self.hyperparams.num_epochs):
            self._handler._arguments.set_status(epoch=epoch)
            self._run_training_epoch(train_loader, valid_loader)

        self._handler.on_training_end()
        self._handler._arguments.reset_status()
        return self._handler.trainer_proxy.get_score_history()

    def _run_training_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> torch.Tensor:
        self._handler.on_training_epoch_begin()
        self._model.train()

        for batch, (features, y_truth) in enumerate(train_loader):
            self._handler._arguments.set_status(batch=batch)
            self._run_training_step(
                x=features.to(device=self.device, dtype=self.xtype),
                y=y_truth.to(device=self.device, dtype=self.ytype),
            )

        self._model.scheduler_step()
        # TODO: self.status.current_epoch returns None here
        # This works: self._handler.hparams.status.current_batch
        # This fails: self.status.current_batch
        if valid_loader is not None and (self.status.current_epoch + 1) % \
                self.hyperparams.num_epochs_per_validation == 0:
            self._run_validating(valid_loader)

        self._handler.on_training_epoch_end()

    def _run_training_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        self._handler.on_training_step_begin()

        self._model.optimizer_zero_grad()
        y_pred, loss = self._model.forward_pass(
            x=x, y=y,
            batch_idx=self.status.current_batch
        )
        self._loss_tracker.update(loss.detach().item())
        loss.backward()
        self._model.optimizer_step()

        with torch.no_grad():
            self._handler.on_training_step_end(x=x, y=y, y_pred=y_pred)

        return y_pred.detach()

    def _run_evaluating(
        self,
        # trainer_arguments: TrainingArguments,
        eval_loader: Optional[torch.utils.data.DataLoader],
    ) -> torch.Tensor:

        self._handler.on_evaluation_run_begin()
        self._model.eval()

        preds = []
        with torch.no_grad():
            for batch, (features, y_truth) in enumerate(eval_loader):
                pred = self._run_evaluating_step(
                    x=features.to(device=self.device, dtype=self.xtype),
                    y=y_truth.to(device=self.device, dtype=self.ytype),
                )
                preds.extend(pred)

        self._handler.on_evaluation_run_end()
        return preds

    def _run_evaluating_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        self._handler.on_evaluation_step_begin()

        y_pred, loss = self._model.forward_pass(
            x=x, y=y, batch_idx=self.status.current_batch
        )
        self._loss_tracker.update(loss.detach().item())

        self._handler.on_evaluation_step_end(x=x, y=y, y_pred=y_pred)

        return y_pred.detach()

    def _run_validating(
        self,
        # trainer_arguments: TrainingArguments,
        valid_loader: Optional[torch.utils.data.DataLoader],
    ) -> torch.Tensor:

        self._handler.on_validation_run_begin()
        self._model.eval()

        with torch.no_grad():
            for batch, (features, y_truth) in enumerate(valid_loader):
                self._run_validating_step(
                    x=features.to(device=self.device, dtype=self.xtype),
                    y=y_truth.to(device=self.device, dtype=self.ytype),
                )

        self._handler.on_validation_run_end()

    def _run_validating_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        self._handler.on_validation_step_begin()

        y_pred, loss = self._model.forward_pass(
            x=x, y=y, batch_idx=self.status.current_batch
        )
        self._loss_tracker.update(loss.detach().item())

        self._handler.on_validation_step_end(x=x, y=y, y_pred=y_pred)

        return y_pred.detach()
