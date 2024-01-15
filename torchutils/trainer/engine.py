#!/usr/bin/env python3
import os
import time
import torch
import logging
import typing
import warnings

from .profilers import ProfilerIterator
from ..models import TrainerModel
from ..datasets import TrainerDataset
from ..metrics import TrainerAverageScore, ScoreContainer, ScoreLogger
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
        'metrics',
        'callbacks',
        'handlers',
        'logger'
    ]

    def __init__(
        self,
        model: TrainerModel,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: typing.Optional[torch.utils.data.Dataset] = None,
        train_dataloader_kwargs: dict = dict(),
        valid_dataloader_kwargs: dict = dict(),
    ):
        self.model: TrainerModel = model
        self.train_dataset: TrainerDataset = TrainerDataset(train_dataset, **train_dataloader_kwargs)
        self.valid_dataset: TrainerDataset = TrainerDataset(valid_dataset, **valid_dataloader_kwargs)
        
        self.metrics: typing.Set[str] = set()
        self.callbacks: typing.List[TrainerCallback] = list()
        self.handlers: typing.Iterable[logging.Handler] = list()
        self.logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)

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
        

    def _initialize_loaders(self, batch_size: int, nepv: int, hparams: dict, profile=False, **kwargs):
        device = self.model.device

        if self.valid_dataset.dataset is None:
            nepv = 0
            valid_dataloader = []
        else:
            valid_dataloader = self.valid_dataset.dataloader(device=device, train=False, **kwargs)
            
        dataloader = self.train_dataset.dataloader(batch_size=batch_size, device=device, train=True, **kwargs)

        if profile:
            score = TrainerAverageScore('load_time', reset_on='epoch_end')
            dataloader = ProfilerIterator(dataloader, score=score)
            if valid_dataloader != []:
                valid_dataloader = ProfilerIterator(valid_dataloader, score=score)

        hparams = dict(**hparams,
                       batch_size=batch_size,
                       num_epochs_per_validation=nepv,
                       model_device=self.model.device)
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
        # use compiled components
        metrics.update(self.metrics)
        callbacks += self.callbacks
        handlers += self.handlers

        # initialize components
        callb_lst = CallbackHandler(callbacks=callbacks)
        score_lst = ScoreContainer(score_names=metrics)
        score_lgr = ScoreLogger()
        score_lgr.add_scores(score_lst)

        # dataloader prepare
        step_timer = TrainerAverageScore('Exec Time', reset_on='epoch_end')
        hparams, trainloader, validloader = self._initialize_loaders(
                batch_size=batch_size, hparams=hparams,
                profile=True, nepv=num_epochs_per_validation)

        # handlers
        self.add_handlers(handlers)
        self.model.add_handlers(handlers)
        callb_lst.add_handlers(handlers)
        score_lgr.add_handlers(handlers)
        callb_lst.attach_score_dict(score_lst.score_dict())

        # logging initialization
        self.logger.info('Training Starts...')
        self.logger.info(str(self.model))
        self.logger.info(str(hparams))

        try:
            callb_lst.on_initialization()

            callb_lst.on_training_begin(hparams)

            for epoch in range(num_epochs):
                # Epoch preperation
                callb_lst.on_training_epoch_begin()

                self.model.train()
                for index, batch in enumerate(trainloader):

                    # Step preperation
                    callb_lst.on_training_step_begin()

                    # Step execution
                    begin_time = time.time()
                    output = self.model.forward_pass(batch_idx=index, batch=batch)
                    self.model.backward_pass()
                    step_timer.update(time.time() - begin_time)

                    # Step finishing
                    # finish computing train metrics ##
                    score_lst.save_values_to_score_dict()
                    score_lgr.log(TRAIN_STEP, epoch_index=epoch, batch_index=index)
                    callb_lst.on_training_step_end(batch_index=index, batch=batch, batch_output=output)
                    score_lst.reset_scores_on_step_end.trigger()
                    del batch, output, begin_time

                # Epoch finishing
                self.model.scheduler_step(epoch_idx=epoch)
                score_lst.save_averages_to_score_dict()
                score_lgr.log(TRAIN_EPOCH, epoch_index=epoch)
                callb_lst.on_training_epoch_end()
                score_lst.reset_scores_on_epoch_end.trigger()

                if hparams['num_epochs_per_validation'] <= 0 \
                        or (epoch + 1) % hparams['num_epochs_per_validation']:
                    continue

                self.model.eval()
                with torch.no_grad():
                    callb_lst.on_validation_run_begin()

                    for index, batch in enumerate(validloader):

                        # Step Preperation
                        callb_lst.on_validation_step_begin()

                        # Step execution
                        begin_time = time.time()
                        output = self.model.forward_pass_no_grad(batch_idx=index, batch=batch)
                        self.model.reset_backward()
                        step_timer.update(time.time() - begin_time)

                        # Step finishing
                        # finish computing valid metrics ##
                        score_lst.save_values_to_score_dict()
                        score_lgr.log(VALID_STEP, epoch_index=epoch, batch_index=index)
                        callb_lst.on_validation_step_end(batch_index=index, batch=batch, batch_output=output)
                        score_lst.reset_scores_on_step_end.trigger()
                        del batch, output, begin_time

                    # Epoch finishing
                    score_lst.save_averages_to_score_dict()
                    score_lgr.log(VALID_RUN, epoch_index=epoch)
                    callb_lst.on_validation_run_end()
                    score_lst.reset_scores_on_epoch_end.trigger()
            callb_lst.on_training_end()
        except StopTrainingException:
            callb_lst.on_stop_training_error()
        finally:
            callb_lst.on_termination()

        self.remove_handlers(handlers)
        self.model.remove_handlers(handlers)
        callb_lst.remove_handlers(handlers)
        score_lgr.remove_handlers(handlers)
        callb_lst.detach_score_dict()


    @torch.no_grad()
    def predict(
        self,
        *test_datasets: torch.utils.data.Dataset,
        metrics: typing.Set[str] = set(),
        callbacks: typing.List[TrainerCallback] = list(),
        handlers: typing.Iterable[logging.Handler] = list(),
        dataloader_kwargs: dict = dict(),
        profile: bool = False,
        **hparams
    ):
        # use compiled components
        metrics.update(self.metrics)
        if profile:
            metrics.add('exec_time')
            metrics.add('load_time')
        callbacks += self.callbacks
        handlers += self.handlers

        # initialize components
        callb_lst = CallbackHandler(callbacks=callbacks)
        score_lst = ScoreContainer(score_names=metrics)
        score_lgr = ScoreLogger()
        score_lgr.add_scores(score_lst)
        exec_timer = TrainerAverageScore('Exec Time', reset_on='epoch_end')
        
        # handlers
        self.add_handlers(handlers)
        self.model.add_handlers(handlers)
        callb_lst.add_handlers(handlers)
        score_lgr.add_handlers(handlers)
        callb_lst.attach_score_dict(score_lst.score_dict())

        # logging initialization
        self.logger.info('Prediction Starts...')
        self.logger.info(str(self.model))
        self.logger.info(str(hparams))

        for test_nu, test_dataset in enumerate(test_datasets):
            if not isinstance(test_dataset, TrainerDataset):
                test_dataset = TrainerDataset(test_dataset)

            dataloader_kwargs.setdefault('batch_size', len(test_dataset.dataset))
            dataloader_kwargs['train'] = False
            dataloader_kwargs['device'] = self.model.device
            dataloader = test_dataset.dataloader(**dataloader_kwargs)
            if profile:
                dataloader = ProfilerIterator(dataloader,
                                              score=TrainerAverageScore('Load Time',
                                                                 reset_on='epoch_end'))

            try:
                callb_lst.on_initialization()

                self.model.eval()
                callb_lst.on_evaluation_run_begin(hparams)

                for index, batch in enumerate(dataloader):

                    # Step Preperation
                    callb_lst.on_evaluation_step_begin()

                    # Step execution
                    begin_time = time.time()
                    output = self.model.forward_pass_no_grad(batch_idx=index, batch=batch)
                    self.model.reset_backward()
                    exec_timer.update(time.time() - begin_time)

                    # Step finishing
                    # finish computing valid metrics ##
                    score_lst.save_values_to_score_dict()
                    score_lgr.log(EVAL_STEP, test_nu, index)
                    callb_lst.on_evaluation_step_end(batch_index=index, batch=batch, batch_output=output)
                    score_lst.reset_scores_on_step_end.trigger()
                    del batch, output, begin_time

                # Epoch finishing
                score_lst.save_averages_to_score_dict()
                score_lgr.log(EVAL_RUN, test_nu, index)
                callb_lst.on_evaluation_run_end()
                score_lst.reset_scores_on_epoch_end.trigger()
                    
            except StopTrainingException:
                callb_lst.on_stop_training_error()
            finally:
                callb_lst.on_termination()

        self.remove_handlers(handlers)
        self.model.remove_handlers(handlers)
        callb_lst.remove_handlers(handlers)
        score_lgr.remove_handlers(handlers)
        callb_lst.detach_score_dict()

    def add_handlers(self, hdlrs: typing.List[logging.Handler]):
        for hdlr in hdlrs:
            self.logger.addHandler(hdlr)

    def remove_handlers(self, hdlrs: typing.List[logging.Handler]):
        for hdlr in hdlrs:
            self.logger.removeHandler(hdlr)
