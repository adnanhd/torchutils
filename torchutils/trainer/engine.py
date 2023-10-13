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
from ..callbacks import CallbackHandler, StopTraining
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

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        metrics: typing.Set[str] = set(),
        handlers: typing.Iterable[logging.Handler] = list(),
        **hparams
    ):
        callback = CallbackHandler([])
        logger = logging.getLogger(self.__class__.__name__)
        mhandler = MetricHandler()
        timer = AverageScore('Duration', reset_on='epoch_end')

        for hdlr in handlers:
            logger.addHandler(hdlr)
            mhandler.logger.addHandler(hdlr)

        try:
            callback.on_initialization()
            cuda = self.model.device.type == 'cuda'
            dataloader = self.train_dataset.dataloader(batch_size=batch_size, cuda=cuda, train=True)
            valid_dataloader = None
            if self.valid_dataset.dataset is not None:
                valid_dataloader = self.valid_dataset.dataloader(cuda=cuda, train=False)
            del cuda

            callback.on_training_begin()

            for epoch in range(num_epochs):
                # Epoch preperation
                callback.on_training_epoch_begin()

                for index, batch in enumerate(dataloader):
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

                if valid_dataloader is None:
                    continue

                with torch.no_grad():
                    callback.on_validation_run_begin()
                    for index, batch in enumerate(valid_dataloader):
                        # Step Preperation
                        callback.on_training_step_begin()

                        # Step execution
                        begin_time = time.time()
                        output = self.model.forward_pass(batch_idx=index, batch=batch)
                        self.model.reset_backward()
                        timer.update(time.time() - begin_time)

                        # Step finishing
                        # finish computing valid metrics ##
                        mhandler.log_values(VALID_STEP, epoch, index)
                        callback.on_training_step_end(batch_index=index, batch=batch, batch_output=output)
                        mhandler.reset_on_step_end.trigger()
                        del batch, output, begin_time

                    # Epoch finishing
                    mhandler.log_averages(VALID_RUN, epoch, index)
                    callback.on_validation_run_end()
                    mhandler.reset_on_epoch_end.trigger()
        except StopTraining:
            callback.on_stop_training_error()
        finally:
            callback.on_termination()
