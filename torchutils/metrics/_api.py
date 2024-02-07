import inspect
import functools
import typing


METRICS_DICT = {}


def wrap_output_dict(metric) -> typing.Dict[str, float]:
    @functools.wraps(metric)
    def wrapped_metric(batch_output, batch_target, **batch_extra_kwds):
        return {metric.__name__: metric(batch_output, batch_target, **batch_extra_kwds)}
    return wrapped_metric


def wrap_numpy_metric(metric):
    @functools.wraps(metric)
    def wrapped_metric(batch_output, batch_target, **batch_extra_kwds):
        return metric(y_pred=batch_output, y_true=batch_target)
    return wrapped_metric


def wrap_tensor_metric(metric):
    @functools.wraps(metric)
    def wrapped_metric(batch_output, batch_target, **batch_extra_kwds):
        return metric(input=batch_output, target=batch_target)
    return wrapped_metric


def register_metric(fn):
    def get_name(name):
        def get_func(func):
            assert name not in METRICS_DICT.keys()
            METRICS_DICT[name] = func
            return func
        return get_func

    if inspect.isfunction(fn):
        return get_name(fn.__name__)(fn)
    elif isinstance(fn, str):
        return get_name(fn)
    else:
        raise TypeError('No support for type ' + str(type(fn)))