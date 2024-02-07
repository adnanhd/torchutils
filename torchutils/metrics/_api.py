import inspect


METRICS_DICT = {}


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