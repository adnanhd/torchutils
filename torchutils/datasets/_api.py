DATASETS_DICT = dict()
from .datasets import WrapDataset
from torch.utils.data import Dataset


def wrap_builder_output(builder):
    maybe_wrap = lambda dset: WrapDataset(dset) if isinstance(dset, Dataset) else dset

    def wrapped_builder(**kwds):
        return tuple(map(maybe_wrap, builder(**kwds)))
    return wrapped_builder


def register_builder(name):
    assert name not in DATASETS_DICT.keys(), f'{name} is already registered'
    def decorator(builder):
        DATASETS_DICT[name] = builder

        def wrapper(**kwds):
            return builder(**kwds)
        return wrapper
    return decorator


def build_dataset(name, **builder_kwds):
    assert name in DATASETS_DICT.keys(), f'{name} is not registered'
    maybe_wrap = lambda dset: WrapDataset(dset) if isinstance(dset, Dataset) else dset
    return tuple(map(maybe_wrap, DATASETS_DICT[name](**builder_kwds)))
