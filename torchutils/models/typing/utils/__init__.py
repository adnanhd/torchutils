from .funcs import obtain_registered_kwargs
from .valids import _BaseValidator
def reverse_dict(d: dict) -> dict:
    assert isinstance(d, dict)
    if d.__len__() == 0: return dict()
    return dict(map(lambda k, v: (v, k), *zip(*d.items())))