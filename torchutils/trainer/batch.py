import torch
import typing
import pydantic
from ..utils.pydantic.types import NpTorchType


class IterationBatch(pydantic.BaseModel):
    # @TODO: create validator for those types
    # xtype: str
    # ytype: str
    # device: str
    input: typing.Optional[NpTorchType]
    preds: typing.Optional[NpTorchType]
    target: typing.Optional[NpTorchType]

    class Config:
        allow_mutation = False

    def collate_fn(self, input, preds, target):
        self.Config.allow_mutation = True
        self.input = input
        self.target = target
        self.preds = preds
        self.Config.allow_mutation = False
