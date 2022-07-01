import torch
import hashlib
import numpy as np
import warnings

from torchutils.utils import hybridmethod, digest_numpy, Version
from sklearn.model_selection import train_test_split
from typing import Callable, Tuple
import torchvision.transforms as tf


# TODO: rename NumpyDataset
class Dataset(torch.utils.data.dataset.Dataset):
    __class_version__ = Version('1.0.0')
    __slots__ = ('x', 'y', 'xtransforms', 'ytransforms')

    def __init__(self, features, labels):
        if len(features) != len(labels):
            raise AssertionError(
                'features and labels expected to be same size but '
                f'found {features.__len__()} != {labels.__len__()}'
            )

        self.x = features
        self.y = labels
        self.xtransforms = tf.Compose([])
        self.ytransforms = tf.Compose([])
        super().__init__()

    @property
    def features(self):
        warnings.warn(
            "features attribute is depricated please use x instead", DeprecationWarning)
        return self.x

    @property
    def labels(self):
        warnings.warn(
            "lables attribute is depricated please use y instead", DeprecationWarning)
        return self.y

    def __hash__(self) -> int:
        return int(self.checksum, 16)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[np.ndarray]:
        return self.xtransforms(self.x[index]), self.ytransforms(self.y[index])

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.__len__()} samples>" \
               f"(features: {self.x.shape}, labels: {self.y.shape})"

    @property
    def checksum(self) -> str:
        digest = f"{digest_numpy(self.x)}(@_@){digest_numpy(self.y)}"
        return hashlib.shake_128(bytes(digest, 'utf-8')).hexdigest(16)

    def _split(self, test_size, **kwargs):
        x_train, X_test, y_train, y_test = train_test_split(
            self.x, self.y,
            test_size=test_size, **kwargs)
        self.x = x_train
        self.y = y_train

        return Dataset(features=X_test, labels=y_test)

    def split(self, test_size, valid_size=None, *args, **kwargs):
        test_dataset = self._split(test_size, *args, **kwargs)
        if valid_size is None:
            return test_dataset
        valid_dataset = self._split(valid_size, *args, **kwargs)
        return test_dataset, valid_dataset

    def apply(self,
              transform: Callable[[torch.Tensor], torch.Tensor] = None,
              label_transform: Callable[[torch.Tensor], torch.Tensor] = None) -> None:
        if transform is not None:
            self.xtransforms.transforms.apply(transform)
        if label_transform is not None:
            self.ytransforms.transforms.apply(label_transform)

    @hybridmethod
    @classmethod
    def load(cls, path: str):
        data = np.load(path)
        print("Imported checksum is", data['checksum'])
        print("Imported version is", data['version'])
        return cls(data['features'], data['labels'])

    @load.instancemethod
    def load(self, path: str):
        data = np.load(path)
        self.x = data['features']
        self.y = data['labels']
        print("Imported checksum is", data['checksum'])
        print("Imported version is", data['version'])

    def save(self, path: str):
        np.savez(path, features=self.x, labels=self.y,
                 checksum=self.checksum, version=str(self.__class_version__))

    def to(self, xtype=None, ytype=None):
        if xtype is not None:
            self.x = self.x.astype(xtype)
        if ytype is not None:
            self.y = self.y.astype(ytype)

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
