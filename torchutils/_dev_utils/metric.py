from ..utils import FunctionalType
import numpy as np
import inspect
import typing
import torch
import functools
import pydantic
from .meter import MeterHandler
import itertools
from .wrappers import wrap_dict_on_return, log_score_on_return, to_numpy_input_tensors, detach_input_tensors, to_cpu_input_tensors


class MetricType(FunctionalType):
    @classmethod
    def __get_validators__(cls):
        yield cls.field_validator
        yield cls.field_wrapper
        yield cls.field_signature_validator

    @classmethod
    def field_signature_validator(cls, field_type, info):
        assert inspect.isfunction(field_type)
        signature = inspect.signature(field_type)
        try:
            assert signature.parameters['batch_output'].annotation == torch.Tensor, 'batch_output must be a torch.Tensor'
            assert signature.parameters['batch_target'].annotation == torch.Tensor, 'batch_target must be a torch.Tensor'
            assert signature.return_annotation == float
        except KeyError as e:
            raise ValueError(e)
        
        return log_score_on_return(field_type)
    
    @classmethod
    def field_wrapper(cls, field_type, info):
        return detach_input_tensors(field_type)


class TorchMetricType(MetricType):
    @classmethod
    def field_wrapper(cls, field_type, info):
        signature = inspect.signature(field_type)
        try:
            assert signature.parameters['output'].annotation == torch.Tensor
            assert signature.parameters['target'].annotation == torch.Tensor
            assert signature.return_annotation == torch.Tensor
        except KeyError as e:
            raise ValueError(e)

        @log_score_on_return
        # @functools.wraps(field_type)
        def wrapped_fn(batch_output: torch.Tensor, batch_target: torch.Tensor, **kwds) -> typing.Set[str, float]:
            return field_type(output=batch_output, target=batch_target, **kwds)
        
        return wrapped_fn


class NumpyMetricType(MetricType):
    @classmethod
    def field_wrapper(cls, field_type, info):
        signature = inspect.signature(field_type)
        try:
            assert signature.parameters['y_pred'].annotation == np.ndarray
            assert signature.parameters['y_true'].annotation == np.ndarray
            assert signature.return_annotation == np.ndarray
        except KeyError as e:
            raise ValueError(e)

        @log_score_on_return
        @to_cpu_input_tensors
        @to_numpy_input_tensors
        # TODO: @functools.wraps(field_type)
        def wrapped_fn(batch_output: torch.Tensor, batch_target: torch.Tensor, **kwds) -> typing.Dict[str, float]:
            return field_type(y_pred=batch_output, y_true=batch_target, **kwds)
        
        return wrapped_fn


TorchMetric = typing.Union[NumpyMetricType, TorchMetricType, MetricType]


class MetricHandler(pydantic.BaseModel):
    metrics: typing.Set[TorchMetric] = set()
    collate_fn: typing.Callable = lambda batch, batch_output: dict(batch_output=batch_output, batch_target=batch[1])
    _handler: MeterHandler = pydantic.PrivateAttr(default_factory=MeterHandler)

    def compute(self, batch, batch_output):
        metric_kwargs = self.collate_fn(batch=batch, batch_output=batch_output)
        proloque_functor = lambda metric: metric(**metric_kwargs).items()
        return dict(itertools.chain.from_iterable(map(proloque_functor, self.metrics)))
            