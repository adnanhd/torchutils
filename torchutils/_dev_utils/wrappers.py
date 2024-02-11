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
    ffn = functools.wraps(fn)(lambda *args, **kwds: {fn.__name__: fn(*args, **kwds)})
    return ffn


def log_score_return_dict(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwds):
        score_dict = fn(*args, **kwds)
        for name, score in score_dict.items():
            # TODO: optimize
            if not AverageMeter.has_instance(name):
                AverageMeter(name=name)
            AverageMeter.get_instance(name).update(score)
        return score_dict
    return wrapped_fn


detach_input_tensors = make_epiloque_functor(torch.Tensor.detach)
to_cpu_input_tensors = make_epiloque_functor(torch.Tensor.cpu)
to_cuda_input_tensors = make_epiloque_functor(torch.Tensor.cuda)
to_numpy_input_tensors = make_epiloque_functor(torch.Tensor.numpy)
from_numpy_input_arrays = make_epiloque_functor(torch.from_numpy, cls=np.ndarray)
