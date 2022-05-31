#!/usr/bin/env python3
import torch
from .valid import _run_validating
from torchutils.utils import profile
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

from ..callbacks import StopTrainingError


def _add_prefix(prefix: str, kwargs: Mapping[str, Any]):
    return {f'{prefix}_{key}': value for key, value in kwargs.items()}


@profile
def _run_training(
    trainer,
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: Optional[torch.utils.data.DataLoader] = None,
    **kwargs,
):
    _UNROLLING_N = 8
    trainer.__handle__(
        "on_training_begin",
        num_epochs=num_epochs,
        batch_size=train_loader.batch_size,
        step_size=train_loader.__len__()
    )

    try:
        for epoch in range(0, num_epochs - _UNROLLING_N + 1, _UNROLLING_N):
            _run_training_epoch(
                trainer, epoch+0, train_loader, valid_loader, **kwargs)
            _run_training_epoch(
                trainer, epoch+1, train_loader, valid_loader, **kwargs)
            _run_training_epoch(
                trainer, epoch+2, train_loader, valid_loader, **kwargs)
            _run_training_epoch(
                trainer, epoch+3, train_loader, valid_loader, **kwargs)
            _run_training_epoch(
                trainer, epoch+4, train_loader, valid_loader, **kwargs)
            _run_training_epoch(
                trainer, epoch+5, train_loader, valid_loader, **kwargs)
            _run_training_epoch(
                trainer, epoch+6, train_loader, valid_loader, **kwargs)
            _run_training_epoch(
                trainer, epoch+7, train_loader, valid_loader, **kwargs)

        for epoch in range((num_epochs // _UNROLLING_N) * _UNROLLING_N, num_epochs):
            _run_training_epoch(trainer, epoch, train_loader,
                                valid_loader, **kwargs)
    except StopTrainingError:
        trainer.__handle__("on_stop_training_error")

    trainer.__handle__("on_training_end", last_epoch=num_epochs)


@profile
def _run_training_epoch(
    trainer,
    epoch: int,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: Optional[torch.utils.data.DataLoader] = None,
    **kwargs,
) -> torch.Tensor:
    trainer.metrics.init(
        epoch=epoch,
        batch_size=train_loader.batch_size,
        step_size=train_loader.__len__()
    )
    
    trainer.__handle__("on_training_epoch_begin", epoch=epoch)

    trainer.model.train()
    train_loss = 0
    for batch, (features, y_truth) in enumerate(train_loader):
        if not torch.is_tensor(features):
            features = torch.cat(features, dim=-1)
        if not torch.is_tensor(y_truth):
            y_truth = torch.cat(y_truth, dim=-1)
        train_loss += _run_training_step(
            trainer=trainer, epoch=epoch, batch_idx=batch, **kwargs,
            x=features.to(device=trainer.device, dtype=trainer.xtype),
            y=y_truth.to(device=trainer.device, dtype=trainer.ytype),
        )
    train_loss /= len(train_loader)

    trainer.metrics.update(epoch)
    results = trainer.metrics.updated_values(epoch)
    results.setdefault('loss', train_loss)
    trainer.__handle__("on_training_epoch_end", epoch=epoch, **results)

    if valid_loader is not None:
        val_results = _run_validating(
            trainer,
            valid_loader,
            **kwargs,
        )

        results.update(_add_prefix('val', val_results))

    trainer.__handle__("on_training_valid_end", epoch=epoch, **results)


def _run_training_step(
    trainer,
    epoch: int,
    batch_idx: int,
    x: torch.Tensor,
    y: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    trainer.__handle__("on_training_step_begin",
                       epoch=epoch,
                       batch=batch_idx)

    trainer.optimizer.zero_grad()

    y_pred = trainer.model(x)
    loss = trainer.criterion(y_pred, y)

    loss.backward()
    trainer.optimizer.step()

    trainer.metrics.step(
        batch_idx=batch_idx,
        y_true=y.detach(),
        y_pred=y_pred.detach(),
    )

    trainer.__handle__("on_training_step_end",
                       epoch=epoch,
                       batch=batch_idx,
                       loss=loss.detach(),
                       batch_output=y_pred.detach(),
                       **trainer.metrics.stepped_values(batch_idx))

    return loss.detach()
