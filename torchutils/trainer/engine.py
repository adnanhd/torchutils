#!/usr/bin/env python3
import os
import time
import torch
import logging
import typing
import warnings

from ..utils import CircularIteratorProfiler, Profiler
from ..models import TrainerModel
from .._dev_utils import AverageMeter, MeterHandler
from ..callbacks import CallbackHandler, StopTrainingException, TrainerCallback

from ..datasets._api import maybe_wrap
from ..datasets import (
    Dataset as TrainerDataset,
    Iterable as TrainerIterable,
)

from torch.utils.data import (
    Dataset as TorchDataset,
    IterableDataset as TorchIterable,
    DataLoader as TorchDataLoader
)

class Trainer:
    __slots__ = [
        'model',
        'logger',
        'metrics',
        'handlers',
        'callbacks',
    ]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def __init__(
        self,
        model: TrainerModel,
        log_level: int = logging.INFO
    ):
        self.model: TrainerModel = model
        self.metrics: typing.Set[str] = set()
        self.callbacks: typing.List[TrainerCallback] = list()
        self.handlers: typing.Iterable[logging.Handler] = list()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

    def compile(
            self,
            metrics: typing.Optional[typing.Set[str]] = None,
            callbacks: typing.Optional[typing.List[TrainerCallback]] = None,
            handlers: typing.Optional[typing.Iterable[logging.Handler]] = None
    ):
        if metrics is not None:
            assert isinstance(metrics, set)
            self.metrics = metrics

        if callbacks is not None:
            assert isinstance(callbacks, list)
            self.callbacks = callbacks

        if handlers is not None:
            assert isinstance(handlers, list)
            self.handlers = handlers

    def _prep_dataset(
            self,
            dataset: typing.Union[torch.utils.data.Dataset, torch.utils.data.DataLoader, None],
            train: bool,
            **dataloader_kwargs
    ):
        if dataset is None or isinstance(dataset, TorchDataLoader):
            self.logger.debug(f'Received {type(dataset)} as Datsaet')
            return dataset            
        elif isinstance(dataset, (TorchDataset, TrainerDataset)):
            dataloader_kwargs.setdefault('batch_size', len(dataset))
        elif isinstance(dataset, (TorchIterable, TrainerIterable)):
            # TODO: make sure that iterable has __len__
            assert 'batch_size' in dataloader_kwargs.keys()

        return maybe_wrap(dataset=dataset).dataloader(train=train, device=self.model.device, **dataloader_kwargs)

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: typing.Optional[torch.utils.data.Dataset] = None,
        train_dataloader_kwargs: dict = dict(),
        valid_dataloader_kwargs: dict = dict(),
        metrics: typing.Set[str] = set(),
        callbacks: typing.List[TrainerCallback] = list(),
        handlers: typing.Iterable[logging.Handler] = list(),
        num_epochs_per_validation: int = 1,
        **hparams
    ):
        # type checking
        assert train_dataset is not None

        # prepare dataloaders
        trainloader = self._prep_dataset(train_dataset, train=True, batch_size=batch_size, **train_dataloader_kwargs)
        validloader = self._prep_dataset(valid_dataset, train=False, **valid_dataloader_kwargs)
        hparams['num_epochs_per_validation'] = num_epochs_per_validation if validloader is not None else 0

        # prepare profilers
        exec_timer = Profiler(name='exec_time')

        # use compiled components
        metrics.update(self.metrics)
        callbacks += self.callbacks
        handlers += self.handlers

        # initialize components
        callb_lst = CallbackHandler(callbacks=callbacks)
        score_lst = MeterHandler()

        # handlers
        for hdlr in handlers:
            self.logger.addHandler(hdlr)
            self.model.add_handler(hdlr)
            callb_lst.add_handler(hdlr)

        # logging initialization
        self.logger.info('Training Starts...')
        self.logger.info(str(self.model))
        self.logger.info(f'train_params: {hparams}')

        try:
            callb_lst.on_initialization()

            callb_lst.on_training_begin(hparams)

            for epoch in range(num_epochs):
                # Epoch preperation
                callb_lst.on_training_epoch_begin(epoch_index=epoch)

                self.model.train()
                for index, batch in enumerate(trainloader):

                    # Step preperation
                    callb_lst.on_training_step_begin(batch_index=index)

                    # Step execution
                    exec_timer.set()
                    output = self.model.forward_pass_on_training_step(batch_idx=index, batch=batch)
                    self.model.backward_pass()
                    exec_timer.lap()

                    # Step finishing
                    # finish computing train metrics ##
                    callb_lst.on_training_step_end(batch_index=index)
                    del index, batch, output

                # Epoch finishing
                self.model.scheduler_step(epoch_idx=epoch)
                callb_lst.on_training_epoch_end()
                score_lst.reset_meters()

                if hparams['num_epochs_per_validation'] <= 0 \
                        or (epoch + 1) % hparams['num_epochs_per_validation']:
                    continue

                self.model.eval()
                with torch.no_grad():
                    callb_lst.on_validation_run_begin(epoch_index=epoch)

                    for index, batch in enumerate(validloader):

                        # Step Preperation
                        callb_lst.on_validation_step_begin(batch_index=index)

                        # Step execution
                        exec_timer.set()
                        output = self.model.forward_pass_on_validation_step(batch_idx=index, batch=batch)
                        self.model.reset_backward()
                        exec_timer.lap()

                        # Step finishing
                        # finish computing valid metrics ##
                        callb_lst.on_validation_step_end(batch_index=index)
                        del index, batch, output

                    # Epoch finishing
                    callb_lst.on_validation_run_end()
                    score_lst.reset_meters()
            callb_lst.on_training_end()
        except StopTrainingException:
            callb_lst.on_stop_training_error()
        finally:
            callb_lst.on_termination()

        
        for hdlr in handlers:
            self.logger.removeHandler(hdlr)
            self.model.remove_handler(hdlr)
            callb_lst.remove_handler(hdlr)

    @torch.no_grad()
    def predict(
        self,
        test_dataset: torch.utils.data.Dataset,
        metrics: typing.Set[str] = set(),
        callbacks: typing.List[TrainerCallback] = list(),
        handlers: typing.Iterable[logging.Handler] = list(),
        dataloader_kwargs: dict = dict(),
        profile: bool = False,
        **hparams
    ):
        # type checking
        assert test_dataset is not None

        # use compiled components
        metrics.update(self.metrics)
        if profile:
            metrics.add('exec_time')
            metrics.add('load_time')
        callbacks += self.callbacks
        handlers += self.handlers

        # initialize components
        callb_lst = CallbackHandler(callbacks=callbacks)
        score_lst = MeterHandler()
        exec_timer = Profiler(name='exec_time')
        
        # handlers
        for hdlr in handlers:
            self.logger.addHandler(hdlr)
            self.model.add_handler(hdlr)
            callb_lst.add_handler(hdlr)

        # logging initialization
        self.logger.info('Prediction Starts...')
        self.logger.info("predict_params: " + str(hparams))

        with torch.no_grad():
            dataloader = self._prep_dataset(test_dataset, train=False, **dataloader_kwargs)

            try:
                callb_lst.on_initialization()

                self.model.eval()
                callb_lst.on_evaluation_run_begin(hparams)

                for index, batch in enumerate(dataloader):

                    # Step Preperation
                    callb_lst.on_evaluation_step_begin(batch_index=index)

                    # Step execution
                    exec_timer.set()
                    output = self.model.forward_pass_on_evauluation_step(batch_idx=index, batch=batch)
                    self.model.reset_backward()
                    exec_timer.lap()

                    # Step finishing
                    # finish computing valid metrics ##
                    callb_lst.on_evaluation_step_end(batch_index=index)
                    del index, batch, output

                # Epoch finishing
                callb_lst.on_evaluation_run_end()
                score_lst.reset_meters()
                    
            except StopTrainingException:
                callb_lst.on_stop_training_error()
            finally:
                callb_lst.on_termination()

        for hdlr in handlers:
            self.logger.removeHandler(hdlr)
            self.model.remove_handler(hdlr)
            callb_lst.remove_handler(hdlr)

    def add_handler(self, *hdlrs: logging.Handler):
        for hdlr in hdlrs:
            self.logger.addHandler(hdlr)

    def remove_handler(self, *hdlrs: logging.Handler):
        for hdlr in hdlrs:
            self.logger.removeHandler(hdlr)
