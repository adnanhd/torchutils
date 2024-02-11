import torch


def args_torch_to_dtype(dtype):
    def decorator(fn):
        def wrapper(*args, **kwds):
            return fn(*map(lambda t: t.to(dtype=dtype), args), **kwds)
        return wrapper
    return decorator


def args_torch_to_device(device):
    def decorator(fn):
        def wrapper(*args, **kwds):
            return fn(*map(lambda t: t.to(device=device), args), **kwds)
        return wrapper
    return decorator


def args_torch_to_detach(fn):
    def wrapper(*args, **kwds):
        return fn(*map(lambda t: t.detach(), args), **kwds)
    return wrapper


def args_numpy_to_tensor(fn):
    def wrapper(*args, **kwds):
        return fn(*map(lambda t: torch.from_numpy(t), args), **kwds)
    return wrapper


def args_tensor_to_numpy(fn):
    def wrapper(*args, **kwds):
        return fn(*map(lambda t: t.cpu().numpy(), args), **kwds)
    return wrapper


def args_to_flatten(fn):
    def wrapper(*args, **kwds):
        return fn(*map(lambda t: t.flatten(), args), **kwds)
    return wrapper


@args_tensor_to_numpy
@args_numpy_to_tensor
@args_to_flatten
def intersection_over_union(output, target):
    return -(output - target)


iou = args_torch_to_detach(intersection_over_union)
