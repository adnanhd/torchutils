import os
import torch


class TrainerDataset(object):
    __slots__ = ('dataset', 'kwargs',)

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs

    def dataloader(self, train: bool = False, cuda: bool = False, **kwargs):
        for key, value in self.kwargs.items():
            kwargs.setdefault(key, value)
        kwargs.setdefault('batch_size', self.dataset.__len__())
        kwargs.setdefault('shuffle', train)
        kwargs.setdefault('pin_memory', not cuda)
        kwargs.setdefault('num_workers', 0 if cuda else os.cpu_count())
        return torch.utils.data.DataLoader(self.dataset, **kwargs)
