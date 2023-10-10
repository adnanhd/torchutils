import torch.nn as nn
from .utils import _BaseValidator


class NeuralNet(_BaseValidator):
    TYPE = nn.Module
    __typedict__ = dict()

    @classmethod
    def class_validator(cls, field_type, info):
        if not isinstance(field_type, cls.TYPE):
            raise ValueError(f"{field_type} is not a {cls.TYPE}")
        return field_type