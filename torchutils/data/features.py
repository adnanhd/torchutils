from collections import OrderedDict
from ..utils import Hashable


class Features(OrderedDict, Hashable):

    def __init__(self, **features):
        super().__init__(features)

    @classmethod
    def new(cls, f):
        if isinstance(f, cls):
            return f
        elif not isinstance(f, dict):
            try:  # if iterable
                f = {str(i): feature
                     for i, feature in enumerate(f)}
            except TypeError:  # if not
                f = {str(0): f}
        return cls(**f)

    def validate(self, example: "Features"):
        assert example.keys() == self.keys()
        def verify_shape(key): return self[key].shape == example[key].shape
        def verify_dtype(key): return self[key].dtype == example[key].dtype
        return all(filter(verify_shape, filter(verify_dtype, self.keys())))

    def reshape(self, example: dict):
        example = OrderedDict(example)
        for key, value in self.items():
            example[key] = example[key] \
                .reshape(value.shape)   \
                .astype(value.dtype)
        return example

    def __hash__(self):
        return int(self.md5, 16)

    def __repr__(self):
        __features_repr__ = ", ".join(f'{key}={value}'
                                      for key, value in self.items())
        return f'{self.__class__.__name__}({__features_repr__})'


class TensorFeatures(Features):
    pass


class InFileFeatures(Features):
    pass
