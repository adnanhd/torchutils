import pydantic
import logging
import typing
import abc
import typing
import numpy as np
import torch
import functools
import inspect
from ._api import METRICS_DICT


class _BaseMetric(abc.ABC):
    @classmethod
    def __get_validators__(cls):
        yield cls.get_registered_metric
        yield cls.validate_metric

    @classmethod
    def get_registered_metric(cls, value, info):
        if isinstance(value, str):
            return METRICS_DICT[value]
        return value

    @classmethod
    def validate_metric(cls, value, info):
        assert inspect.isfunction(value), f'{value} is function'
        return value



def create_metric_validator(**params):
    def validator(cls, value):
        value_signature = inspect.signature(value).parameters
        __args_str__ = ", ".join(params.keys())
        assert inspect.isfunction(value), f'{value} is function'
        assert set(value_signature.keys()).issuperset(params.keys()), f'{__args_str__} in the argument list'
        for pname, ptype in params.items():
            assert value_signature[pname].annotation == ptype, f'{pname} is of type {ptype}'
        return value
    return validator


numpy_metric_validator = create_metric_validator(y_pred=np.ndarray, y_true=np.ndarray)
tensor_metric_validator = create_metric_validator(input=torch.Tensor, target=torch.Tensor)


class NumpyMetric(_BaseMetric):
    @classmethod
    def __get_validator__(cls):
        yield numpy_metric_validator


class TensorMetric(_BaseMetric):
    @classmethod
    def __get_validator__(cls):
        yield tensor_metric_validator