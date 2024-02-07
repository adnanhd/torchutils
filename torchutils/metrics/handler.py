from .._dev_utils import TrainerBaseModel
from .metric import _BaseMetric
from itertools import chain
import typing
import pydantic
import torch


class MetricHandler(pydantic.BaseModel):
    metrics: typing.Set[_BaseMetric]

    def compute(self, batch_output: torch.Tensor, batch_target: torch.Tensor, **batch_extra_kwds) -> typing.Dict[str, float]:
        def call(fn) -> typing.Dict[str, float]:
            return fn(batch_output=batch_output, batch_target=batch_target, **batch_extra_kwds)
        return dict(chain.from_iterable(map(dict.items, map(call, self.metrics))))