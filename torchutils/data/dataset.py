import torch
import hashlib
import numpy as np

from .utils import hybridmethod
from sklearn.model_selection import train_test_split
from typing import Callable


__version__ = '2.0.a'
# TODO: keep in mind that Dataset might be shuffled
# there must be two dataset RandomDataset, SequentialDataset
# RandomDataset must reorder its elements according
# to its randomization


class Dataset(torch.utils.data.dataset.Dataset):
    __slots__ = ('features', 'labels')

    def __init__(self, features, labels, transform=None):
        assert len(features) == len(
            labels), f'features and labels expected to be same size but found {features.__len__()} != {labels.__len__()}'
        self.labels = labels
        self.features = features
        super().__init__()

    def _split(self, test_size, **kwargs):
        x_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels,
            test_size=test_size, **kwargs)
        self.features = x_train
        self.labels = y_train

        return Dataset(features=X_test, labels=y_test)

    def split(self, test_size, valid_size=None, *args, **kwargs):
        test_dataset = self._split(test_size, *args, **kwargs)
        if valid_size is None:
            return test_dataset
        valid_dataset = self._split(valid_size, *args, **kwargs)
        return test_dataset, valid_dataset

    def apply(self, fn: Callable):
        ...

    def filter(self, fn: Callable):
        ...

    def load(self, path: str):
        ...

    def save(self, path: str):
        ...
    
    def to(self, xtype=None, ytype=None):
        if xtype is not None: self.features = self.features.astype(xtype)
        if ytype is not None: self.labels = self.labels.astype(ytype)

    #def __hash__(self):
    #    return self.features.__hash__() ^ \
    #        self.labels.__hash__() ^ \
    #        self.metadata.__hash__() ^ \
    #        self.__version__.__hash__()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index: int):
        feature = self.features[index]
        label = self.labels[index]
        return feature, label

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.__len__()} samples>(features: {self.features.shape}, labels: {self.labels.shape})"

    @hybridmethod
    def dataloader(cls, features, labels, batch_size: int = None, train: bool = True, transform=None, **kwargs):
        if batch_size is None:
            batch_size = len(features)
        return torch.utils.data.DataLoader(
            cls(features, labels, transform=transform),
            batch_size=batch_size, shuffle=train, **kwargs)

    @dataloader.instancemethod
    def dataloader(self, batch_size: int = None, train: bool = True, **kwargs):
        if batch_size is None:
            batch_size = self.__len__()
        kwargs['shuffle'] = train
        return torch.utils.data.DataLoader(self, **kwargs, batch_size=batch_size)


