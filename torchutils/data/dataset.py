import torch
import hashlib

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
        """
        self._metadata = Builder.generate_metadata()
        self._features = Builder.generate_features()
        self._labels = Builder.generate_features()
        self.feature_func self.label_func <-- remove
        self.transform, self.collator <-- don't remove
        # TODO: check what is collator used for?
        """
        self.labels = labels
        self.features = features
        super().__init__()

    def _split(self, test_size, **kwargs):
        x_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels,
            test_size=test_size, **kwargs)
        self.features = x_train
        self.labels = y_train

        return Dataset(features=X_test, labels=y_test,
                       transform=self.transform)

    def split(self, test_size, valid_size=None, *args, **kwargs):
        test_dataset = self.split(test_size, *args, **kwargs)
        if valid_size is None:
            return test_dataset
        valid_dataset = self.split(valid_size, *args, **kwargs)
        return test_dataset, valid_dataset

    def apply(self, fn: Callable):
        pass

    def filter(self, fn: Callable):
        pass

    def load(self, path: str):
        pass

    def save(self, path: str):
        pass

    def __hash__(self):
        return self.features.__hash__() ^ \
            self.lables.__hash__() ^ \
            self.metadata.__hash__() ^ \
            self.__version__.__hash__()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index: int):
        feature = self.features[index]
        label = self.labels[index]
        return list(feature), list(label)

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.__len__()} samples>(features: {self.metadata.features}, labels: {self.metadata.labels})"

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
        return torch.utils.data.DataLoader(self, **kwargs, shuffle=train, batch_size=batch_size)
