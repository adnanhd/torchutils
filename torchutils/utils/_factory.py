import numpy as np
import functools
import inspect
import torch


def torch_fn_parameters_wrapper(f, cls=torch.Tensor):
    def maybe_f(x):
        return f(x) if isinstance(x, cls) else x

    def decorator(fn):
        @functools.wraps(fn)
        def decorate_function(*args, **kwds):
            return fn(*map(maybe_f, args),
                      **dict(zip(kwds.keys(), map(maybe_f, kwds.values()))))
        return decorate_function
    return decorator


detach_input_tensors = torch_fn_parameters_wrapper(torch.Tensor.detach)
to_cpu_input_tensors = torch_fn_parameters_wrapper(torch.Tensor.cpu)
to_cuda_input_tensors = torch_fn_parameters_wrapper(torch.Tensor.cuda)
to_numpy_input_tensors = torch_fn_parameters_wrapper(torch.Tensor.numpy)
from_numpy_input_arrays = torch_fn_parameters_wrapper(torch.from_numpy, cls=np.ndarray)
