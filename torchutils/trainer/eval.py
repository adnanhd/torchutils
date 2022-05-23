#!/usr/bin/env python3
import torch
from .utils import profile
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
def _run_evaluating(
    trainer,
    loader: torch.utils.data.DataLoader,
    *args,
    **kwargs,
) -> torch.Tensor:
    trainer.metrics.init(batch_size=loader.batch_size, step_size=loader.__len__())
    trainer.metrics.reset()
    trainer.__handle__(
        "on_evaluation_run_begin",
        batch_size=loader.batch_size,
        step_size=loader.__len__()
    )

    trainer.model.eval()
    with torch.no_grad():
        preds = [
            _run_evaluating_step(
                trainer=trainer, batch_idx=batch, *args, **kwargs,
                x=features.to(device=trainer.device, dtype=trainer.xtype),
                y=y_true.to(device=trainer.device, dtype=trainer.ytype),
            )
            for batch, (features, y_true) in enumerate(loader)
        ]

    trainer.metrics.update()
    trainer.__handle__(
        "on_evaluation_run_end",
        last_batch=len(preds)
    )

    return torch.concat(preds)


def _run_evaluating_step(
    trainer,
    batch_idx: int,
    x: torch.Tensor,
    y: torch.Tensor,
    **kwargs,
):
    trainer.__handle__("on_evaluation_step_begin", step=batch_idx)

    y_pred = trainer.model(x)
    loss = trainer.criterion(y_pred, y)

    trainer.metrics.step(
        batch_idx=batch_idx,
        y_true=y.detach(),
        y_pred=y_pred.detach(),
    )

    trainer.__handle__("on_evaluation_step_end",
                       batch=batch_idx,
                       loss=loss.detach(),
                       batch_output=y_pred.detach(),
                       **trainer.metrics.stepped_values(batch_idx))

    return y_pred.detach()
