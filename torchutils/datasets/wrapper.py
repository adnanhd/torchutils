import os
import torch


class TrainerDataset(object):
    __slots__ = ('dataset', 'kwargs',)
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs

    def dataloader(self, batch_size: int = None, train: bool = False, cuda: bool = False, **kwargs):
        if batch_size is None:
            batch_size = self.dataset.__len__()
        for key, value in self.kwargs.items():
            kwargs.setdefault(key, value)
        kwargs.setdefault('shuffle', train)
        kwargs.setdefault('pin_memory', not cuda)
        kwargs.setdefault('num_workers',0 if cuda else os.cpu_count())
        return torch.utils.data.DataLoader(self.dataset, **kwargs, batch_size=batch_size)