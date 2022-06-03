#!/usr/bin/env python3
import torch
from .valid import _run_validating
from torchutils.callbacks import StopTrainingError
from torchutils.utils.pydantic import TrainerArguments, HandlerArguments
from torchutils.utils import profile
Trainer = "Trainer"
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


@profile
def _run_training(
    trainer: Trainer,
    trainer_arguments: TrainerArguments,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: Optional[torch.utils.data.DataLoader] = None,
):
    _UNROLLING_N = 8
    trainer._handler.on_training_begin()

    try:
        for epoch in range(0, trainer_arguments.num_epochs - _UNROLLING_N + 1, _UNROLLING_N):
            _run_training_epoch(trainer, train_loader, valid_loader)
            _run_training_epoch(trainer, train_loader, valid_loader)
            _run_training_epoch(trainer, train_loader, valid_loader)
            _run_training_epoch(trainer, train_loader, valid_loader)
            _run_training_epoch(trainer, train_loader, valid_loader)
            _run_training_epoch(trainer, train_loader, valid_loader)
            _run_training_epoch(trainer, train_loader, valid_loader)
            _run_training_epoch(trainer, train_loader, valid_loader)

        for epoch in range((trainer_arguments.num_epochs // _UNROLLING_N) * _UNROLLING_N, trainer_arguments.num_epochs):
            _run_training_epoch(trainer, train_loader, valid_loader)
    except StopTrainingError:
        trainer.on_stop_training_error()

    trainer._handler.on_training_end()

@profile
def _run_training_epoch(
    trainer: Trainer,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: Optional[torch.utils.data.DataLoader] = None,
) -> torch.Tensor:

    trainer._handler.on_training_epoch_begin()
    trainer._model.train()
    trainer._handler.reset()
    
    for batch, (features, y_truth) in enumerate(train_loader):
        if not torch.is_tensor(features): features = torch.cat(features, dim=-1)
        if not torch.is_tensor(y_truth):  y_truth = torch.cat(y_truth, dim=-1)
        _run_training_step(trainer=trainer,
            x=features.to(device=trainer.device, dtype=trainer.xtype),
            y=y_truth.to(device=trainer.device, dtype=trainer.ytype),
        )

    trainer._handler.on_training_epoch_end()

    if valid_loader is not None:
        val_results = _run_validating(trainer, valid_loader)


def _run_training_step(
    trainer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    trainer._handler.on_training_step_begin()

    y_pred, loss = trainer._model.attached_step(x=x, y=y, batch_idx=trainer._status.current_batch)
    trainer._handler.update(batch_loss=loss.detach())

    with torch.no_grad():
        trainer._handler.on_training_step_end(x=x, y=y, y_pred=y_pred)
