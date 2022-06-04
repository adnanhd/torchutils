#!/usr/bin/env python3
import torch
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

#profile
def _run_evaluating(
    trainer: "Trainer",
    #trainer_arguments: TrainingArguments,
    valid_loader: Optional[torch.utils.data.DataLoader],
) -> torch.Tensor:

    trainer._handler.on_evaluation_run_begin()
    trainer._model.eval()
    trainer._handler.reset()
    
    with torch.no_grad():
        for batch, (features, y_truth) in enumerate(valid_loader):
            if not torch.is_tensor(features): features = torch.cat(features, dim=-1)
            if not torch.is_tensor(y_truth):  y_truth = torch.cat(y_truth, dim=-1)
            _run_evaluating_step(trainer=trainer,
                x=features.to(device=trainer.device, dtype=trainer.xtype),
                y=y_truth.to(device=trainer.device, dtype=trainer.ytype),
            )

    trainer._handler.on_evaluation_run_end()


def _run_evaluating_step(
    trainer: "Trainer",
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    trainer._handler.on_evaluation_step_begin()

    y_pred, loss = trainer._model.detached_step(x=x, y=y, batch_idx=trainer.status.current_batch)
    trainer._handler.update(batch_loss=loss.detach())

    trainer._handler.on_evaluation_step_end(x=x, y=y, y_pred=y_pred)
