import abc
import numpy as np
from torch import from_numpy


def _obtain_shape(dim: tuple, arr_shape: tuple) -> tuple:
    if dim is None:
        dim = arr_shape
    cardin = arr_shape.__len__()  # cardinality of the array
    dim = tuple(d if d != -1 else (cardin + d) for d in dim)
    return tuple(1 if d in dim else arr_shape[d] for d in range(cardin))


class Transform(abc.ABC):
    def __init__(self, data: np.ndarray):
        self._original = data.copy()
        self._reference = data

    @abc.abstractmethod
    def normalize_arr(self, arr):
        ...

    @abc.abstractmethod
    def denormalize_arr(self, arr):
        ...

    @property
    def normalized_data(self):
        return self.normalize_arr(self._original)

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

    def __call__(self, arr):
        return self.denormalize_arr(arr)

    def __enter__(self):
        self.normalize()
        return self

    def __exit__(self, *args):
        self.denormalize()
        return self


class MinMaxTransform(Transform):
    def __init__(self, data: np.ndarray, axis: tuple = None):
        super().__init__(data)
        self._min = data.min(axis=axis)
        self._max = data.max(axis=axis)
        self._min = self._min.reshape(_obtain_shape(axis, data.shape))
        self._max = self._max.reshape(_obtain_shape(axis, data.shape))

    def to_tensor(self, device):
        self._min = from_numpy(self._min).to(device=device)
        self._max = from_numpy(self._max).to(device=device)

    def normalize_arr(self, arr):
        return (arr - self._min) / (self._max - self._min)

    def denormalize_arr(self, arr):
        return arr * (self._max - self._min) + self._min


class MeanStdTransform(Transform):
    def __init__(self, data: np.ndarray, axis: tuple = None):
        super().__init__(data)
        self._mean = data.mean(axis=axis)
        self._std = data.std(axis=axis)
        self._mean = self._mean.reshape(_obtain_shape(axis, data.shape))
        self._std = self._std.reshape(_obtain_shape(axis, data.shape))

    def to_tensor(self, device):
        from torch import from_numpy
        self._mean = from_numpy(self._mean).to(device=device)
        self._std = from_numpy(self._std).to(device=device)

    def normalize_arr(self, arr):
        return (arr - self._mean) / self._std

    def denormalize_arr(self, arr):
        return arr * self._std + self._mean
