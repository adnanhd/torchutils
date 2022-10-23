import torch
import hashlib
import numpy as np
import warnings
import logging

from sklearn.model_selection import train_test_split
from torchutils.utils import hybridmethod, digest_numpy, Version

import torchvision.transforms as tf
from typing import Tuple, List


def identity(x):
    return x


def true(x):
    return True


def transpose(arr):
    return zip(*arr)


Transforms = object


class NumpyDataset(torch.utils.data.dataset.Dataset):
    __class_version__ = Version('1.1.0')
    __slots__ = ('__dataset__', '__transforms__', '__target_transforms__',
                 '__transforms_flag__', '__target_transforms_flag__')

    def __init__(self,
                 features,
                 labels,
                 transforms: List[Transforms] = [],
                 target_transforms: List[Transforms] = []):
        super().__init__()
        if len(features) != len(labels):
            raise AssertionError(
                'features and labels expected to be same size but '
                f'found {features.__len__()} != {labels.__len__()}'
            )
        self.__dataset__ = tuple(zip(features, labels))
        self.__transforms__ = tf.Compose(transforms)
        self.__target_transforms__ = tf.Compose(target_transforms)
        self.__transforms_flag__ = bool(len(transforms))
        self.__target_transforms_flag__ = bool(len(target_transforms))

    def __len__(self) -> int:
        return len(self.__dataset__)

    def __getitem__(self, index: int) -> Tuple[np.ndarray]:
        x, y = self.__dataset__[index]

        if self.__transforms_flag__:
            x = self.__transforms__(x)

        if self.__target_transforms_flag__:
            y = self.__target_transforms__(y)

        return x, y

    @property
    def x(self) -> str:
        # return (x for x, y in self.__dataset__)
        return map(lambda x_y: x_y[0], self.__dataset__)

    @property
    def y(self) -> str:
        # return (y for x, y in self.__dataset__)
        return map(lambda x_y: x_y[1], self.__dataset__)

    @property
    def features(self) -> np.ndarray:
        return np.array(list(self.x))

    @property
    def labels(self) -> np.ndarray:
        return np.array(list(self.y))

    @property
    def checksum(self) -> str:
        digest = '+' + "@".join(map(digest_numpy, self.x))
        digest = hashlib.shake_128(bytes(digest, 'utf-8'))
        return digest.hexdigest(16)

    @property
    def target_checksum(self) -> str:
        digest = '+' + "@".join(map(digest_numpy, self.y))
        digest = hashlib.shake_128(bytes(digest, 'utf-8'))
        return digest.hexdigest(16)

    def apply(self, fn=identity, target_fn=identity):
        def func(x, y):
            return fn(x), target_fn(y)

        self.__dataset__ = tuple(map(func, *transpose(self.__dataset__)))

    def filter(self, fn=true, target_fn=true):
        def func(pair):
            return fn(pair[0]) and target_fn(pair[1])
        self.__dataset__ = tuple(filter(func, self.__dataset__))

    def __hash__(self) -> int:
        return int(self.checksum, 16) ^ int(self.target_checksum, 16)

    def __repr__(self):
        if self.__dataset__.__len__() == 0:
            info = tuple(), tuple()
        else:
            x, y = self.__dataset__[0]
            info = x.shape, y.shape
        return f"<{self.__class__.__name__}: {self.__len__()} samples>" + \
            "(features: {}, labels: {})".format(*info)

    def new(self, x=[], y=[]):
        transforms = self.__transforms__.transforms[:]
        target_transforms = self.__target_transforms__.transforms[:]
        return self.__class__(x, y, transforms, target_transforms)

    def copy(self):
        dataset = self.new()
        dataset.__dataset__ = self.__dataset__
        return dataset

    def deepcopy(self):
        return self.new(*transpose(self.__dataset__))

    def split(self, test_size, **kwargs):
        splits = train_test_split(*transpose(self.__dataset__),
                                  test_size=test_size, **kwargs)
        self.__dataset__ = tuple(transpose(splits[::2]))
        return self.new(*splits[1::2])

    @ hybridmethod
    @ classmethod
    def load(cls, path: str):
        data = np.load(path, allow_pickle=True)
        logger = logging.getLogger(f'{__name__}.{NumpyDataset.__name__}')
        logger.info(f"Imported dataset version is  {data['version']}")
        logger.info(f"Imported feature checksum is {data['checksum']}")
        logger.info(f"Imported label checksum is {data['target_checksum']}")
        data_version = Version(*data['version'].item().split('.'))
        if data_version < NumpyDataset.__class_version__:
            warnings.warn(f'loaded class {data_version} has an older structure'
                          f' than {NumpyDataset.__class_version__}.')

        def is_transforms(item):
            key, value = item
            return key in ('transforms', 'target_transforms')
        kwargs = dict(filter(is_transforms, data.items()))
        return NumpyDataset(data['x'], data['y'], **kwargs)

    @ load.instancemethod
    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        data_version = Version(*data['version'].item().split('.'))
        if data_version < self.__class__.__class_version__:
            warnings.warn(f'loaded class {data_version} has an older structure'
                          f' than {self.__class__.__class_version__}.')
        self.__dataset__ = transpose((data['x'], data['y']))
        self.__transforms__ = tf.Compose(data['transforms'])
        self.__transforms_flag__ = bool(len(data['transforms']))
        self.__target_transforms__ = tf.Compose(data['target_transforms'])
        self.__target_transforms_flag__ = bool(len(data['target_transforms']))
        logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        logger.info(f"Imported dataset version is  {data['version']}")
        logger.info(f"Imported feature checksum is {data['checksum']}")
        logger.info(f"Imported label checksum is {data['target_checksum']}")

    def save(self, path: str):
        x, y = transpose(self.__dataset__)
        np.savez(path, x=x, y=y,
                 checksum=self.checksum,
                 target_checksum=self.target_checksum,
                 transforms=self.__transforms__.transforms,
                 target_transforms=self.__target_transforms__.transforms,
                 version=str(self.__class_version__))

    @ hybridmethod
    def dataloader(cls, features, labels,
                   batch_size: int = None, train: bool = True,
                   transform: list = [], target_transforms: list = [],
                   **kwargs):
        if batch_size is None:
            batch_size = len(features)
        return torch.utils.data.DataLoader(
            cls(features, labels, transform=transform),
            batch_size=batch_size, shuffle=train, **kwargs)

    @ dataloader.instancemethod
    def dataloader(self, batch_size: int = None, train: bool = True, **kwargs):
        if batch_size is None:
            batch_size = self.__len__()
        kwargs['shuffle'] = train
        kwargs['batch_size'] = batch_size
        return torch.utils.data.DataLoader(self, **kwargs)
