import numpy as np
import functools
import torch
import typing
from .meter import AverageMeter
try:
    import ray
except ImportError:
    ray = None


def wrap_dict_on_return(fn):
    return functools.wraps(fn)(lambda *args, **kwds: {fn.__name__: fn(*args, **kwds)})


def _get_meters_from_name(meter_names: typing.Sequence[str]) -> typing.Dict[str, AverageMeter]:
    meters = dict()
    for name in meter_names:
        if AverageMeter.has_instance(name):
            meters[name] = AverageMeter.get_instance(name)
        else:
            meters[name] = AverageMeter(name=name)
    return meters


def ray_remote_parallelize(fn):
    assert ray is not None, 'make sure you installed ray!'
    if AverageMeter.has_instance(fn.remote.__name__):
        meter = AverageMeter.get_instance(fn.remote.__name__)
    else:
        meter = AverageMeter(name=fn.remote.__name__)
    
    @functools.wraps(fn.remote)
    def wrapped_fn(batch_output: torch.Tensor, batch_target: torch.Tensor):
        scores = []
        remote_objs = [fn.remote(*arg) for arg in zip(batch_output, batch_target)]
        for score in ray.get(remote_objs):
            meter.update(score, n=1)
            scores.append(meter.value)
        return sum(scores) / len(scores)
    return wrapped_fn


def log_score_return_dict(fn):
    if AverageMeter.has_instance(fn.__name__):
        meter = AverageMeter.get_instance(fn.__name__)
    else:
        meter = AverageMeter(name=fn.__name__)

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwds) -> typing.Dict[str, float]:
        score = fn(*args, **kwds)
        meter.update(score)
        return {fn.__name__: score}
    return wrapped_fn


def log_scores_return_dict(*meter_names: str):
    meters = _get_meters_from_name(meter_names)
    
    @functools.wraps(log_score_return_dict)
    def _log_score_return_dict(fn):
        @functools.wraps(fn)
        def wrapped_fn(*args, **kwds):
            score_dict = fn(*args, **kwds)
            for name, score in score_dict.items():
                meters[name].update(score)
            return score_dict
        return wrapped_fn
    return _log_score_return_dict


def make_epiloque_functor(f, cls=torch.Tensor):
    def maybe_f(x):
        return f(x) if isinstance(x, cls) else x

    def decorator(fn):
        @functools.wraps(fn)
        def decorate_function(*args, **kwds):
            return fn(*map(maybe_f, args),
                      **dict(zip(kwds.keys(), map(maybe_f, kwds.values()))))
        return decorate_function
    return decorator


detach_input_tensors = make_epiloque_functor(torch.Tensor.detach)
to_cpu_input_tensors = make_epiloque_functor(torch.Tensor.cpu)
to_cuda_input_tensors = make_epiloque_functor(torch.Tensor.cuda)
to_numpy_input_tensors = make_epiloque_functor(torch.Tensor.numpy)
from_numpy_input_arrays = make_epiloque_functor(torch.from_numpy, cls=np.ndarray)
