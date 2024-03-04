from ..utils import FunctionalType
import types
from abc import ABCMeta
import inspect
import typing
import torch
import pydantic
from .meter import MeterHandler, AverageMeter
from itertools import chain
from .decorators import (
    wrap_dict_on_return,
    # log_score_return_dict,
    ray_remote_parallelize,
    # to_numpy_input_tensors,
    detach_input_tensors,
    # to_cpu_input_tensors
)


class MetricType(FunctionalType):
    @classmethod
    def __get_validators__(cls):
        yield cls.get_function_from_name
        yield MetricType.field_validator
        yield cls.field_signature_validator
        yield cls.field_wrapper

    @classmethod
    def field_validator(cls, field_type, info):
        if isinstance(field_type, torch.jit.ScriptFunction):
            def foo(batch_output: torch.Tensor, batch_target: torch.Tensor
                    ) -> typing.Dict[str, float]:
                return field_type(batch_output=batch_output,
                                  batch_target=batch_target)
            foo.__name__ = field_type.name
            return foo
        if ray is not None and isinstance(field_type,
                                          ray.remote_function.RemoteFunction):
            return ray_remote_parallelize(field_type)
        elif isinstance(field_type, FunctionalType):
            return field_type
        raise ValueError('unexpected type')

    @classmethod
    def field_signature_validator(cls, field_type, info):
        # Type Check
        assert inspect.isfunction(field_type), 'field_signature_validator'

        signature = inspect.signature(field_type)
        expected = dict(batch_output=torch.Tensor,
                        batch_target=torch.Tensor)

        # Argument Type Check
        for name, annot in expected.items():
            if name not in signature.parameters.keys():
                raise ValueError(f'{name} must be an argument')
            if signature.parameters[name].annotation != annot:
                raise AssertionError(f'{name} must be of {annot}')

        # Return Type Check
        if signature.return_annotation == float:
            return wrap_dict_on_return(field_type)
        elif signature.return_annotation == typing.Dict[str, float]:
            return field_type
        else:
            raise AssertionError('return type mismatch in '
                                 'field_signature_validator')

    @classmethod
    def field_wrapper(cls, field_type, info):
        return detach_input_tensors(field_type)


class MetricHandler(pydantic.BaseModel):
    metrics: typing.Set[MetricType] = set()
    collate_fn: typing.Callable = (
            lambda batch, batch_output: dict(batch_output=batch_output,
                                             batch_target=batch[1])
    )
    _handler: MeterHandler = pydantic.PrivateAttr(default_factory=MeterHandler)

    def compute(self, batch, batch_output):
        metric_kwargs = self.collate_fn(batch=batch, batch_output=batch_output)
        metric_dicts = map(lambda m: m(**metric_kwargs).items(), self.metrics)
        for name, score in chain.from_iterable(metric_dicts):
            if not AverageMeter.has_instance(name):
                AverageMeter(name=name)
            AverageMeter.get_instance(name).update(score)


# registary
ABCMeta.register(MetricType, types.FunctionType)
ABCMeta.register(MetricType, torch.jit.ScriptFunction)

try:
    import ray
    ABCMeta.register(MetricType, ray.remote_function.RemoteFunction)
except ImportError:
    ray = None
