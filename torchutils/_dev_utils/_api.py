import inspect
import torch
import typing
import numpy as np
from .metric import MetricType
from .decorators import log_score_return_dict, to_cpu_input_tensors, detach_input_tensors, to_numpy_input_tensors


def register_metric(fn):
    return MetricType.register(fn)


def register_torch_metric(fn):
    assert inspect.isfunction(fn), 'register_torch_metric'
    signature = inspect.signature(fn)
    expected = dict(output=torch.Tensor,
                    target=torch.Tensor)

    for name, annot in expected.items():
        if name not in signature.parameters.keys():
            raise ValueError(f'{name} must be an argument')
        if signature.parameters[name].annotation != annot:
            raise AssertionError(f'{name} must be of {annot}')
    assert signature.return_annotation == float

    # TODO: Fix workaround solution
    #@wraps(fn)
    def wrapped_fn(batch_output: torch.Tensor,
                    batch_target: torch.Tensor,
                    **kwds) -> typing.Dict[str, float]:
        return fn(output=batch_output, target=batch_target, **kwds)
    
    wrapped_fn.__name__ = fn.__name__

    if signature.return_annotation == float:
        wrapped_fn = log_score_return_dict(wrapped_fn)

    MetricType.register(wrapped_fn)

    return fn

def register_numpy_metric(fn):
    assert inspect.isfunction(fn), 'register_numpy_metric'
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
    # TODO: Fix workaround solution
    #@wraps(fn)
    def wrapped_fn(batch_output: torch.Tensor,
                    batch_target: torch.Tensor,
                    **kwds) -> typing.Dict[str, float]:
        return fn(y_pred=batch_output, y_true=batch_target, **kwds)
    
    wrapped_fn.__name__ = fn.__name__

    if signature.return_annotation == float:
        wrapped_fn = log_score_return_dict(wrapped_fn)

    MetricType.register(wrapped_fn)

    return fn