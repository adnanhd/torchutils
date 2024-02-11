from ..utils import FunctionalType
import numpy as np
import inspect
import typing
import torch
from functools import wraps, reduce
import pydantic
from .meter import MeterHandler
from itertools import chain
from .wrappers import (
    wrap_dict_on_return,
    log_score_return_dict,
    to_numpy_input_tensors,
    detach_input_tensors,
    to_cpu_input_tensors
)


def composite_function(*func):
    def compose(f, g):
        return lambda x: f(g(x))

    return reduce(compose, func, lambda x: x)


class MetricType(FunctionalType):
    @classmethod
    def __get_validators__(cls):
        yield cls.get_function_from_name
        yield cls.field_validator
        yield cls.field_signature_validator
        yield cls.field_wrapper

    @classmethod
    def field_signature_validator(cls, field_type, info):
        assert inspect.isfunction(field_type)
        signature = inspect.signature(field_type)
        expected = dict(batch_output=torch.Tensor,
                        batch_target=torch.Tensor)

        for name, annot in expected.items():
            if name not in signature.parameters.keys():
                raise ValueError(f'{name} must be an argument')
            if signature.parameters[name].annotation != annot:
                raise AssertionError(f'{name} must be of {annot}')
        if signature.return_annotation == float:
            field_type = wrap_dict_on_return(field_type)
        else:
            assert signature.return_annotation == typing.Dict[str, float]

        return field_type

    @classmethod
    def field_wrapper(cls, field_type, info):
        wrapper = composite_function(log_score_return_dict,
                                     detach_input_tensors)
        return wrapper(field_type)

    @classmethod
    def register_torch_metric(cls, fn):
        assert inspect.isfunction(fn)
        signature = inspect.signature(fn)
        expected = dict(output=torch.Tensor,
                        target=torch.Tensor)

        for name, annot in expected.items():
            if name not in signature.parameters.keys():
                raise ValueError(f'{name} must be an argument')
            if signature.parameters[name].annotation != annot:
                raise AssertionError(f'{name} must be of {annot}')
        assert signature.return_annotation == float

        @wraps(fn)
        def wrapped_fn(batch_output: torch.Tensor,
                       batch_target: torch.Tensor,
                       **kwds) -> typing.Set[str, float]:
            return fn(output=batch_output, target=batch_target, **kwds)

        cls.register(wrapped_fn)

        return fn

    @classmethod
    def register_numpy_metric(cls, fn):
        assert inspect.isfunction(fn)
        signature = inspect.signature(fn)
        expected = dict(y_pred=np.ndarray,
                        y_true=np.ndarray)

        for name, annot in expected.items():
            if name not in signature.parameters.keys():
                raise ValueError(f'{name} must be an argument')
            if signature.parameters[name].annotation != annot:
                raise AssertionError(f'{name} must be of {annot}')
        assert signature.return_annotation == float

        @to_cpu_input_tensors
        @to_numpy_input_tensors
        @wraps(fn)
        def wrapped_fn(batch_output: torch.Tensor,
                       batch_target: torch.Tensor,
                       **kwds) -> typing.Dict[str, float]:
            return fn(y_pred=batch_output, y_true=batch_target, **kwds)

        return cls.register(wrapped_fn)

        return fn


class MetricHandler(pydantic.BaseModel):
    metrics: typing.Set[MetricType] = set()
    collate_fn: typing.Callable = lambda batch, batch_output: dict(batch_output=batch_output, batch_target=batch[1])
    _handler: MeterHandler = pydantic.PrivateAttr(default_factory=MeterHandler)

    def compute(self, batch, batch_output):
        metric_kwargs = self.collate_fn(batch=batch, batch_output=batch_output)
        metric_dicts = map(lambda m: m(**metric_kwargs).items(), self.metrics)
        return dict(chain.from_iterable(metric_dicts))
