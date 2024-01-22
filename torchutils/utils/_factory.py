import torch
import numpy as np
import inspect


def torch_fn_parameters_wrapper(f, cls=torch.Tensor):
    maybe_f = lambda x: f(x) if isinstance(x, cls) else x
    def decorator(fn):
        def wrapped_fn(*args, **kwds):
            return fn(*map(maybe_f, args), **dict(zip(kwds.keys(), map(maybe_f, kwds.values()))))
        return wrapped_fn
    return decorator


detach_input_tensors = torch_fn_parameters_wrapper(torch.Tensor.detach)
to_cpu_input_tensors = torch_fn_parameters_wrapper(torch.Tensor.cpu)
to_cuda_input_tensors = torch_fn_parameters_wrapper(torch.Tensor.cuda)
to_numpy_input_tensors = torch_fn_parameters_wrapper(torch.Tensor.numpy)
from_numpy_input_arrays = torch_fn_parameters_wrapper(torch.from_numpy, cls=np.ndarray)
