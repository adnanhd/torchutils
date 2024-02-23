from .datasets import WrapDataset, WrapIterable, TorchDataset, TorchIterable, _Wrap, Dataset
import functools


DATASETS_DICT = dict()


def maybe_wrap(dataset) -> Dataset:
    if isinstance(dataset, TorchDataset):
        return WrapDataset(dataset=dataset)
    elif isinstance(dataset, TorchIterable):
        return WrapIterable(dataset=dataset)
    return dataset

def is_wrapped(dataset):
    return isinstance(dataset, _Wrap)


def wrap_builder_output(builder):
    def wrapped_builder(**kwds):
        return tuple(map(maybe_wrap, builder(**kwds)))
    return wrapped_builder


def register_builder(name):
    assert name not in DATASETS_DICT.keys(), f'{name} is already registered'
    def decorator(builder):
        DATASETS_DICT[name] = builder

        @functools.wraps(builder)
        def wrapper(**kwds):
            return builder(**kwds)
        return wrapper
    return decorator


def build_dataset(name, **builder_kwds):
    assert name in DATASETS_DICT.keys(), f'{name} is not registered'
    return tuple(map(maybe_wrap, DATASETS_DICT[name](**builder_kwds)))