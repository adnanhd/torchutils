#!/usr/bin/env python3
import os
import time
import torch
import logging
import typing
import warnings

from ..models import TrainerModel
from ..datasets import TrainerDataset
from ..metrics import AverageScore, MetricHandler
from ..callbacks import CallbackHandler, StopTrainingException, TrainerCallback
from ..logging import (
    TRAIN_EPOCH, VALID_RUN, EVAL_RUN,
    TRAIN_STEP, VALID_STEP, EVAL_STEP
)


class Trainer:
    __slots__ = [
        'model',
        'device',
        'train_dataset',
        'valid_dataset',
    ]

    def __init__(
        self,
        model: TrainerModel,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: typing.Optional[torch.utils.data.Dataset] = None,
    ):
        self.model: TrainerModel = model
        self.train_dataset: TrainerDataset = TrainerDataset(train_dataset)
        self.valid_dataset: TrainerDataset = TrainerDataset(valid_dataset)

    def _initialize_loaders(self, batch_size: int, nepv: int, hparams: dict):
        device = self.model.device
        if self.valid_dataset.dataset is None:
            nepv = 0
            valid_dataloader = []
        else:
            valid_dataloader = self.valid_dataset.dataloader(device=device, train=False)
        hparams = dict(**hparams,
                       batch_size=batch_size,
                       num_epochs_per_validation=nepv,
                       model_device=self.model.device,
                       )
        dataloader = self.train_dataset.dataloader(batch_size=batch_size, device=device, train=True)
        return hparams, dataloader, valid_dataloader

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        metrics: typing.Set[str] = set(),
        callbacks: typing.List[TrainerCallback] = list(),
        handlers: typing.Iterable[logging.Handler] = list(),
        num_epochs_per_validation: int = 1,
        **hparams
    ):
        callback = CallbackHandler(callbacks=callbacks)
        mhandler = MetricHandler(metrics=metrics)
        timer = AverageScore('Duration', reset_on='epoch_end')
        hparams, trainloader, validloader = self._initialize_loaders(
                batch_size=batch_size, hparams=hparams,
                nepv=num_epochs_per_validation)

        # handlers
        self.model.add_handlers(handlers)
        callback.add_handlers(handlers)
        mhandler.add_handlers(handlers)

        try:
            callback.on_initialization()

            callback.on_training_begin(hparams)

            for epoch in range(num_epochs):
                # Epoch preperation
                callback.on_training_epoch_begin()

                for index, batch in enumerate(trainloader):
                    # Step preperation
                    callback.on_training_step_begin()

                    # Step execution
                    begin_time = time.time()
                    output = self.model.forward_pass(batch_idx=index, batch=batch)
                    self.model.backward_pass()
                    timer.update(time.time() - begin_time)

                    # Step finishing
                    # finish computing train metrics ##
                    mhandler.log_values(TRAIN_STEP, epoch, index)
                    callback.on_training_step_end(batch_index=index, batch=batch, batch_output=output)
                    mhandler.reset_on_step_end.trigger()
                    del batch, output, begin_time

                # Epoch finishing
                self.model.scheduler_step(epoch_idx=epoch)
                mhandler.log_averages(TRAIN_EPOCH, epoch, index)
                callback.on_training_epoch_end()
                mhandler.reset_on_epoch_end.trigger()

                if hparams['num_epochs_per_validation'] <= 0 \
                        or (epoch + 1) % hparams['num_epochs_per_validation']:
                    continue

                with torch.no_grad():
                    callback.on_validation_run_begin()
                    for index, batch in enumerate(validloader):
                        # Step Preperation
                        callback.on_validation_step_begin()

                        # Step execution
                        begin_time = time.time()
                        output = self.model.forward_pass(batch_idx=index, batch=batch)
                        self.model.reset_backward()
                        timer.update(time.time() - begin_time)

                        # Step finishing
                        # finish computing valid metrics ##
                        mhandler.log_values(VALID_STEP, epoch, index)
                        callback.on_validation_step_end(batch_index=index, batch=batch, batch_output=output)
                        mhandler.reset_on_step_end.trigger()
                        del batch, output, begin_time

                    # Epoch finishing
                    mhandler.log_averages(VALID_RUN, epoch, index)
                    callback.on_validation_run_end()
                    mhandler.reset_on_epoch_end.trigger()
            callback.on_training_end()
        except StopTrainingException:
            callback.on_stop_training_error()
        finally:
            callback.on_termination()

        self.model.remove_handlers(handlers)
        callback.remove_callbacks(handlers)
        mhandler.remove_handlers(handlers)
