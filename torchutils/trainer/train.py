#!/usr/bin/env python3
from typing import (
    List,
    Dict,
    Any,
    Mapping,
    Optional,
    Union,
    Callable,
    Tuple,
    Iterable
)
import torch
from .valid import _run_validating
from torchutils.callbacks import StopTrainingError
from torchutils.utils.pydantic import TrainingArguments, HandlerArguments
from torchutils.utils import profile
Trainer = "Trainer"

# profile


def _run_training(
    trainer: Trainer,
    trainer_arguments: TrainingArguments,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: Optional[torch.utils.data.DataLoader] = None,
):
    trainer._handler.on_training_begin()

    for epoch in range(trainer_arguments.resume_epochs, trainer_arguments.num_epochs):
        trainer._handler.arguments.update_status(epoch=epoch)
        _run_training_epoch(trainer, trainer_arguments,
                            train_loader, valid_loader)

    trainer._handler.on_training_end()
    trainer._handler.arguments.reset_status()

# profile


def _run_training_epoch(
    trainer: Trainer,
    trainer_arguments: TrainingArguments,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: Optional[torch.utils.data.DataLoader] = None,
) -> torch.Tensor:
    trainer._handler.on_training_epoch_begin()
    trainer._model.train()
    trainer._handler.reset()

    for batch, (features, y_truth) in enumerate(train_loader):
        trainer._handler.arguments.update_status(batch=batch)
        #if not torch.is_tensor(features): features = torch.cat(features, dim=-1)
        #if not torch.is_tensor(y_truth):  y_truth = torch.cat(y_truth, dim=-1)
        _run_training_step(trainer=trainer,
                           x=features.to(device=trainer.device,
                                         dtype=trainer.xtype),
                           y=y_truth.to(device=trainer.device,
                                        dtype=trainer.ytype),
                           )

    trainer._handler.on_training_epoch_end()

    # TODO: trainer.status.current_epoch returns None here
    # This works: trainer._handler.arguments.status.current_batch
    # This fails: trainer.status.current_batch
    current_epoch = trainer._handler.arguments.status.current_epoch
    if valid_loader is not None and (current_epoch + 1) % \
            trainer_arguments.num_epochs_per_validation == 0:
        val_results = _run_validating(trainer, valid_loader)


def _run_training_step(
    trainer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    trainer._handler.on_training_step_begin()

    y_pred, loss = trainer._model.attached_step(
        x=x, y=y, batch_idx=trainer.status.current_batch)
    trainer._handler.update(batch_loss=loss.detach())

    with torch.no_grad():
        trainer._handler.on_training_step_end(x=x, y=y, y_pred=y_pred)
