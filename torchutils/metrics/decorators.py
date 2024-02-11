def args_to_detach(fn):
    def wrapper(*args, **kwds):
        return fn(*map(lambda t: t.detach(), args), **kwds)
    return wrapper


def args_to_numpy(fn):
    def wrapper(*args, **kwds):
        return fn(*map(lambda t: t.detach().cpu().numpy(), args), **kwds)
    return wrapper


def args_to_flatten(fn):
    def wrapper(*args, **kwds):
        return fn(*map(lambda t: t.flatten(), args), **kwds)
    return wrapper
