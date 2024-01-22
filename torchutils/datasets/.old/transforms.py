import torch
import numpy


class NumpyToTensor(object):
    def __call__(self, array: numpy.ndarray) -> torch.Tensor:
        assert isinstance(array, numpy.ndarray)
        return torch.from_numpy(array)


class ToDevice(object):
    def __init__(self, device=None, dtype=None):
        assert device is not None or dtype is not None
        self.device = device
        self.dtype = dtype

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        if self.device is not None and self.dtype is not None:
            return tensor.to(device=self.device, dtype=self.dtype)
        elif self.device is not None:
            return tensor.to(device=self.device)
        else:
            return tensor.to(dtype=self.dtype)