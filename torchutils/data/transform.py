import numpy as np
import warnings
import abc

def _obtain_shape(dim: tuple, arr_shape: tuple) -> tuple:
    if dim is None: dim = arr_shape
    cardin = arr_shape.__len__() # cardinality of the array
    dim = tuple(d if d != -1 else (cardin + d) for d in dim)
    return tuple(1 if d in dim else arr_shape[d] for d in range(cardin))


class Transform(abc.ABC):
    def __init__(self, data: np.ndarray):
        self._original = data.copy()
        self._reference = data

    @abc.abstractproperty
    def normalized_data(self):
        ...
    
    @property
    def denormalized_data(self):
        return self._original.copy()

    @property
    def data(self):
        return self._reference.copy()

    def normalize(self):
        self._reference[:] = self.normalized_data

    def denormalize(self):
        self._reference[:] = self.denormalized_data

    def __enter__(self):
        self.normalize()
        return self

    def __exit__(self, *args):
        self.denormalize()
        return self


class MinMaxTransform(Transform):
    def __init__(self, data: np.ndarray, axis: tuple=None):
        super().__init__(data)
        self._min = data.min(axis=axis)
        self._max = data.max(axis=axis)
        self._min = self._min.reshape(_obtain_shape(axis, data.shape))
        self._max = self._max.reshape(_obtain_shape(axis, data.shape))

    @property
    def normalized_data(self):
        return (self._original - self._min) / (self._max - self._min)


class MeanStdTransform(Transform):
    def __init__(self, data: np.ndarray, axis: tuple=None):
        super().__init__(data)
        self._mean = data.mean(axis=axis)
        self._std = data.std(axis=axis)
        self._mean = self._mean.reshape(_obtain_shape(axis, data.shape))
        self._std = self._std.reshape(_obtain_shape(axis, data.shape))

    @property
    def normalized_data(self):
        return (self._original - self._mean) / self._std
