#!/usr/bin/env python3
import os
import time
import torch
import logging
import typing
import warnings
import numpy as np
from typing import (
    Iterable,
    Mapping,
    Optional,
    Union,
)

from ..models import TrainerModel
from ..datasets import TrainerDataset
from ..metrics import AverageScore, MetricHandler
from ..callbacks import CallbackHandler, StopTraining


from .logging import (
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
        valid_dataset: Optional[torch.utils.data.Dataset] = None,
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
        logger.setLevel(15)
        timer = AverageScore('Iteration Duration')
        epoch_timer = AverageScore('Epoch Duration')

        for hdlr in handlers:
            logger.addHandler(hdlr)
            hdlr.setLevel(15)

        try:
            callback.on_initialization()
            dataloader = self.train_dataset.dataloader(batch_size, cuda=self.model.device.type == 'cuda', train=True)
            
            callback.on_training_begin()

            for epoch in range(num_epochs):
                callback.on_training_epoch_begin()
                begin_epoch_time = time.time()

                for index, batch in enumerate(dataloader):
                    callback.on_training_step_begin()
                    begin_time = time.time()
                    output = self.model.forward_pass(batch_idx=index, batch=batch)
                    ## finish computing train metrics ##
                    timer.update(time.time() - begin_time)
                    logger.log(TRAIN_STEP, {"epoch": epoch, 'batch_index': index, **MetricHandler.score_values()})
                    callback.on_training_step_end(batch_index=index, batch=batch, batch_output=output)
                    self.model.backward_pass()

                epoch_timer.update(time.time() - begin_epoch_time)
                logger.log(TRAIN_EPOCH, MetricHandler.score_averages())
                self.model.scheduler_step(epoch_idx=epoch)
                callback.on_training_epoch_end()
                ## reset computed step-metrics ##
                timer.reset()

                with torch.no_grad():
                    callback.on_validation_run_begin()
                    
                    for index, batch in enumerate(dataloader):
                        callback.on_training_step_begin()
                        begin_time = time.time()
                        output = self.model.forward_pass(batch_idx=index, batch=batch)
                        ## finish computing valid metrics ##
                        timer.update(time.time() - begin_time)
                        logger.log(VALID_STEP, {"epoch": epoch, "batch_index": index, **MetricHandler.score_values()} )
                        callback.on_training_step_end(batch_index=index, batch=batch, batch_output=output)
                        
                    logger.log(VALID_RUN, MetricHandler.score_averages())
                    self.model.reset_backward()
                    callback.on_validation_run_end()
                    ## reset computed step-metrics ##
                    timer.reset()
                
        except StopTraining:
            callback.on_stop_training_error()
        finally:
            callback.on_termination()
            epoch_timer.reset()

        
    
