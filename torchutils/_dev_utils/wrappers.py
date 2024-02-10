import numpy as np
import functools
import torch
from .meter import AverageMeter


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


def wrap_dict_on_return(fn):
    return functools.wraps(fn)(lambda *args, **kwds: {fn.__name__: fn(*args, **kwds)})


def log_score_on_return(fn):
    meter = AverageMeter(name=fn.__name__)
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwds):
        score = fn(*args, **kwds)
        meter.update(score)
        return {fn.__name__: score}
    return wrapped_fn


def log_scores_on_return(*names):
    meters = tuple(map(AverageMeter, names))
    def decorator(fn):
        @functools.wraps(fn)
        def wrapped_fn(*args, **kwds):
            scores = fn(*args, **kwds)
            for score, meter in zip(scores, meters):
                meter.update(score)
            return scores
        return wrapped_fn
    return decorator


detach_input_tensors = make_epiloque_functor(torch.Tensor.detach)
to_cpu_input_tensors = make_epiloque_functor(torch.Tensor.cpu)
to_cuda_input_tensors = make_epiloque_functor(torch.Tensor.cuda)
to_numpy_input_tensors = make_epiloque_functor(torch.Tensor.numpy)
from_numpy_input_arrays = make_epiloque_functor(torch.from_numpy, cls=np.ndarray)
