import abc
import copy
import torch
import numpy as np


def _obtain_shape(dim: tuple, arr_shape: tuple) -> tuple:
    if dim is None:
        dim = arr_shape
    cardin = arr_shape.__len__()  # cardinality of the array
    dim = tuple(d if d != -1 else (cardin + d) for d in dim)
    return tuple(1 if d in dim else arr_shape[d] for d in range(cardin))


class Transform(abc.ABC):
    def __init__(self, data: np.ndarray):
        self._original = copy.deepcopy(data)
        self._reference = data

    def __enter__(self):
        self.normalize()
        return self

    def __exit__(self, *args):
        self.denormalize()
        return self

    def __call__(self, arr):
        return self.denormalize_arr(arr)

    @property
    def normalized_data(self):
        return self.normalize_arr(self._original)

    @property
    def initial_data(self):
        return self._original

    @property
    def data(self):
        return self._reference

    @abc.abstractmethod
    def normalize_arr(self, arr):
        ...

    @abc.abstractmethod
    def denormalize_arr(self, arr):
        ...

    def normalize(self):
        self._reference[:] = self.normalized_data

    def denormalize(self):
        self._reference[:] = self.initial_data


class MinMaxTransform(Transform):
    def __init__(self, data: np.ndarray, axis: tuple = None):
        super().__init__(data)
        self._min = data.min(axis=axis)
        self._max = data.max(axis=axis)
        self._min = self._min.reshape(_obtain_shape(axis, data.shape))
        self._max = self._max.reshape(_obtain_shape(axis, data.shape))

    def to_tensor(self, device):
        self._min = torch.from_numpy(self._min).to(device=device)
        self._max = torch.from_numpy(self._max).to(device=device)

    def to_numpy(self):
        self._min = self._min.cpu().detach().numpy()
        self._max = self._max.cpu().detach().numpy()

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
        self._mean = torch.from_numpy(self._mean).to(device=device)
        self._std = torch.from_numpy(self._std).to(device=device)

    def to_numpy(self):
        self._mean = self._mean.cpu().detach().numpy()
        self._std = self._std.cpu().detach().numpy()

    def normalize_arr(self, arr):
        return (arr - self._mean) / self._std

    def denormalize_arr(self, arr):
        return arr * self._std + self._mean
