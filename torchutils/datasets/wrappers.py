import os
import torch


class DataLoaderWrapper(object):
    __slots__ = ('dataset', 'kwargs',)

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs

    def dataloader(self, train: bool = False, device: torch.device = 'cpu', **kwargs):
        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        for key, value in self.kwargs.items():
            kwargs.setdefault(key, value)
        kwargs.setdefault('batch_size', self.dataset.__len__())
        kwargs.setdefault('shuffle', train)
        kwargs.setdefault('pin_memory', device.type != 'cuda')
        kwargs.setdefault('num_workers', (device.type != 'cuda') * os.cpu_count())
        return torch.utils.data.DataLoader(self.dataset, **kwargs)