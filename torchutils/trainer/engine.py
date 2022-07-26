#!/usr/bin/env python3
import os
import torch
import typing
import warnings
import numpy as np
from typing import (
    Iterable,
    Mapping,
    Optional,
    Union,
)


from .handler import TrainingHandler, EvaluatingHandler
from .arguments import TrainingArguments, EvaluatingArguments


from ..models.utils import TrainerModel
from ..data.dataset import Dataset
from ..metrics import (
    MetricHandler,
    AverageMeter
)
from ..logging import (
    LoggerHandler,
    TrainerLogger,
    ExperimentProfiler
)

from ..callbacks import (
    TrainerCallback,
    StopTrainingError,
    CallbackHandler
)


class Trainer:
    __slots__ = [
        '_model',
        '_loggers',
        '_profiler',
        '_metrics',
        '_callbacks',
        # @TODO: migrate xtype and ytype to TrainerModel
        'ytype',
    ]

    def __init__(
        self,
        model: Union[TrainerModel, torch.nn.Module],
        loss: Optional[Union[torch.autograd.Function, torch.nn.Module]] = None,
        device: Optional[Union[str, torch.device]] = None,  # todo: remove
        xtype: Union[torch.dtype, np.dtype, type] = torch.float32,
        ytype: Union[torch.dtype, np.dtype, type] = torch.float32,
    ):
        assert isinstance(model, TrainerModel) or loss is not None
        if not isinstance(model, TrainerModel):
            # @TODO: add optimizer and scheduler and kwargs as
            # arguments to pass it thrrough TrainerModel
            model = TrainerModel(model=model, criterion=loss, optimizer='Adam')

        self._model: TrainerModel = model
        self._model.device = device

        self._model.dtype = xtype
        self.ytype = ytype

        # initializing handlers
        self._metrics = MetricHandler()
        self._callbacks = CallbackHandler()
        self._loggers = LoggerHandler()
        self._profiler = ExperimentProfiler()

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
        assert isinstance(dataset, torch.utils.data.Dataset), \
            "Dataset must be inherited from torch.utils.data.Dataset"

        try:
            if isinstance(dataset.x, np.ndarray) \
                    and self.device != torch.device('cpu'):
                dataset.x = torch.from_numpy(dataset.x)
            dataset.x = dataset.x.to(
                device=self.device,
                dtype=self.xtype
            )
            if isinstance(dataset.y, np.ndarray) and \
                    self.device != torch.device('cpu'):
                dataset.y = torch.from_numpy(dataset.y)
            dataset.y = dataset.y.to(
                device=self.device,
                dtype=self.ytype
            )
        except AttributeError:
            warnings.warn(
                "Using a Dataset not derived from torchutils.data.Dataset "
                "is dangerous for dtype integrity"
            )

        kwargs.setdefault('shuffle', train_mode)
        kwargs.setdefault('pin_memory', not torch.cuda.is_available())
        kwargs.setdefault('num_workers',
                          0 if torch.cuda.is_available() else os.cpu_count())
        if not train_mode or batch_size is None:
            kwargs['batch_size'] = dataset.__len__()
        else:
            kwargs['batch_size'] = batch_size

        if isinstance(dataset, Dataset):
            return dataset.dataloader(**kwargs)
        else:
            return torch.utils.data.DataLoader(dataset, **kwargs)

    def compile_model_and_hparams(self,
                                  ** hparams) -> None:
        for field_name in set(self._model.__fields__):
            hparams.setdefault(field_name, getattr(self._model, field_name))
        self._model = TrainerModel(**hparams)

    def compile_handlers(
            self,
            loggers: typing.Iterable[TrainerLogger] = list(),
            metrics: typing.Iterable[AverageMeter] = list(),
            callbacks: typing.Iterable[TrainerCallback] = list(),
    ):
        self._loggers.add_loggers(loggers)
        self._metrics.add_score_meters(metrics)
        self._callbacks.add_callbacks(callbacks)

    def decompile_handlers(
            self,
            loggers: typing.Iterable[TrainerLogger] = list(),
            callbacks: typing.Iterable[TrainerCallback] = list(),
    ):
        self._loggers.remove_loggers(loggers)
        self._callbacks.remove_callbacks(callbacks)

    def clear_handlers(self):
        self._loggers.clear_loggers()
        self._callbacks.clear_callbacks()

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        train_dataset: torch.utils.data.Dataset,
        learning_rate: float = None,
        valid_dataset: Optional[torch.utils.data.Dataset] = None,
        train_dataloader_kwargs: Optional[Mapping] = dict(),
        valid_dataloader_kwargs: Optional[Mapping] = dict(),
        history: typing.Set[str] = set(),
        **hparams
    ):
        if learning_rate is not None:
            self._model.learning_rate = learning_rate
        else:
            learning_rate = self._model.learning_rate
        hparams['learning_rate'] = learning_rate

        train_dl = self.create_dataloader(
            dataset=train_dataset, train_mode=True,
            batch_size=batch_size, **train_dataloader_kwargs,
        )

        valid_dl = None
        if valid_dataset is not None:
            valid_dl = self.create_dataloader(
                dataset=valid_dataset,
                train_mode=False,
                **valid_dataloader_kwargs,
                batch_size=hparams.setdefault('valid_dl_batch_size', -1)
            )
            hparams['valid_dl_batch_size'] = valid_dl.batch_size

        handler = TrainingHandler(
            metrics=self._metrics,
            loggers=self._loggers,
            callbacks=self._callbacks,
            history=history,
            arguments=TrainingArguments(
                model=self._model,
                train_dl=train_dl,
                valid_dl=valid_dl,
                num_epochs=num_epochs,
                **hparams
            )
        )

        try:
            handler.on_initialization()
            self._run_training(handler)
        except StopTrainingError:
            handler.on_stop_training_error()
        # else:
        #     handler.on_termination()
        finally:
            handler.on_termination()
        return handler.interface.get_stored_scores(*history)

    def evaluate(
        self,
        dataset,
        dataloader_kwargs=dict(),
        history: typing.Set[str] = set(),
        **hparams
    ):
        eval_dl = self.create_dataloader(
            dataset=dataset,
            train_mode=False,
            batch_size=dataset.__len__(),
            **dataloader_kwargs,
        )

        handler = EvaluatingHandler(
            metrics=self._metrics,
            loggers=self._loggers,
            callbacks=self._callbacks,
            history=history,
            arguments=EvaluatingArguments(
                model=self._model,
                eval_dl=eval_dl,
                **hparams
            )
        )

        try:
            handler.on_initialization()
            results = self._run_evaluating(handler)
        except StopTrainingError:
            handler.on_stop_training_error()
        else:
            return results
        finally:
            handler.on_termination()

    def _run_training(
        self,
        handler: TrainingHandler,
    ):
        handler.on_training_begin()

        for epoch_idx in range(handler.hparams.resume_epochs,
                               handler.hparams.num_epochs):
            self._run_training_epoch(epoch_idx, handler)

        handler.on_training_end()

    def _run_training_epoch(
        self,
        epoch_idx: int,
        handler: TrainingHandler,
    ) -> torch.Tensor:
        handler.on_training_epoch_begin(epoch_idx)
        self._model.train()

        for batch_idx, batch in enumerate(handler.hparams.train_dl):
            self._run_training_step(batch_idx, batch, handler)

        handler.on_training_epoch_end()
        self._model.reset_backward()
        if handler.hparams.valid_dl is not None \
                and (handler.interface.status.current_epoch + 1) % \
                handler.interface.hparams.num_epochs_per_validation == 0:
            self._run_validating(handler)

    def _run_training_step(
        self,
        batch_idx: int,
        batch: Iterable[torch.Tensor],
        handler: TrainingHandler,
    ) -> torch.Tensor:
        x = batch[0].to(device=self.device, dtype=self.xtype)
        y = batch[1].to(device=self.device, dtype=self.ytype)

        handler.on_training_step_begin(batch_idx)

        y_pred = self._model.forward_pass(x, y, batch_idx)
        self._model.backward_pass()

        handler.on_training_step_end(x=x, y=y, y_pred=y_pred)

        return y_pred.detach()

    @torch.no_grad()
    def _run_evaluating(
        self,
        handler: EvaluatingHandler
    ) -> torch.Tensor:

        handler.on_evaluation_run_begin()
        self._model.eval()

        preds = []
        for batch_idx, batch in enumerate(handler.hparams.eval_dl):
            pred = self._run_evaluating_step(batch_idx, batch, handler)
            preds.extend(pred)

        self._model.reset_backward()
        handler.on_evaluation_run_end()
        return preds

    @torch.no_grad()
    def _run_evaluating_step(
        self,
        batch_idx: int,
        batch: Iterable[torch.Tensor],
        handler: EvaluatingHandler
    ) -> torch.Tensor:
        x = batch[0].to(device=self.device, dtype=self.xtype)
        y = batch[1].to(device=self.device, dtype=self.ytype)

        handler.on_evaluation_step_begin()

        y_pred = self._model.forward_pass(x=x, y=y, batch_idx=batch_idx)

        handler.on_evaluation_step_end(x=x, y=y, y_pred=y_pred)

        return y_pred.detach()

    @torch.no_grad()
    def _run_validating(
        self,
        handler: TrainingHandler
    ) -> torch.Tensor:

        handler.on_validation_run_begin()
        self._model.eval()

        for step_idx, batch in enumerate(handler.hparams.valid_dl):
            self._run_validating_step(step_idx, batch, handler)

        self._model.reset_backward()
        handler.on_validation_run_end()

    @torch.no_grad()
    def _run_validating_step(
        self,
        batch_idx: int,
        batch: Iterable[torch.Tensor],
        handler: TrainingHandler
    ) -> torch.Tensor:
        x = batch[0].to(device=self.device, dtype=self.xtype)
        y = batch[1].to(device=self.device, dtype=self.ytype)

        handler.on_validation_step_begin()

        y_pred = self._model.forward_pass(x=x, y=y, batch_idx=batch_idx)

        handler.on_validation_step_end(x=x, y=y, y_pred=y_pred)

        return y_pred.detach()
