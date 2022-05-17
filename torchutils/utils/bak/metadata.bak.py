import os
import math
import copy
import torch
import numpy as np

from .dtypes import Datum
from .utils import hybridmethod

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
    
@dataclass
class Value:
    _dtype: Union[type, str, torch.dtype, np.dtype] = field()
    _shape: Optional[Union[  torch.Size,  tuple]] = field(default=None)

    def __init__(self, dtype, shape=None):
        self.dtype = dtype
        self.shape = shape

    @property
    def dtype(self):
        if not isinstance(self._dtype, str):
            return self._dtype
        else:
            assert self._dtype.lower() in _str_types
            return _str_types[self._dtype.lower()]

    @dtype.setter
    def dtype(self, dtype) -> None:
        allowed_dtypes = self.__annotations__['_dtype']._subs_tree()[1:]
        assert any(isinstance(dtype, dt) for dt in allowed_dtypes)
        self._dtype = dtype

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape) -> None:
        allowed_shapes = self.__annotations__['_shape']._subs_tree()[1:]
        assert any(isinstance(shape, sh) for sh in allowed_shapes)
        self._shape = shape

    def check_shape(self, other: "Value") -> bool:
        assert isinstance(other, self.__class__)
        return self.shape == other.shape

    def check_dtype(self, other: "Value") -> bool:
        assert isinstance(other, self.__class__)
        return self.dtype == other.dtype

    def __eq__(self, other: "Value") -> bool:
        assert isinstance(other, self.__class__)
        return self.check_shape(other) and self.check_dtype(other)

    def __repr__(self):
        dtype = _types2str[self._dtype] if isinstance(self._dtype, type) else self._dtype
        return f'{self.__class__.__name__}(dtype={dtype}, shape={self._shape})'

"""
        def _add_checksum(arr):
            return {
                'value': arr, 
                'checksum': {
                    'sha256': hashlib.sha256(bytes(arr)).hexdigest(), 
                    'blake2b': hashlib.blake2b(bytes(arr)).hexdigest(),
                    }
                }
"""

class Features:
    __slots__ = ('_features',)
    def __init__(self, *features):
        self._features = list()

    def __len__(self):
        return self._features.__len__()

    def __getitiem__(self, index):
        return self._features.__getitem__(index)

    def _dump(sel): # -3 dict
        return {}

    def _load(self, features):
        pass # update self._features

    def __hash__(self):
        return None

@dataclass
class Version:
    """
    __slot__ = (
        '_package', # torchutils package 
        '_module',  # torchutils.data module
        '_class',   # torchutils.data.Dataset class
        '_file',    # torchutils.data.dataset file
        '_dataset'  # Dataset() dataset instance
    )
    
    _class = '1.0.0' 
    from torchutils import __version__ as _package
    from torchutils.data import __version__ as _module
    from torchutils.data.dataset import __version__ as _file
    """
    _major: Union[int, str] = field(default=1)
    _minor: Union[int, str] = field(default=0)
    _patch: Union[int, str] = field(default=0)
    def __init__(self, major=1, minor=0, patch=0, version=None):
        if isinstance(major, str):
            version = major

        if isinstance(version, str):
            self.version = version
        elif all(isinstance(v, int) or isinstance(v, str) 
                for v in (major, minor, patch)):
            self._major = major
            self._minor = minor
            self._patch = patch
        else:
            raise AssertionError('Either version or major-minor-patch must be set to str or int accordingly...')

    @property
    def version(self):
        return '{self._major}.{self._minor}.{self._patch}'

    @version.setter
    def version(self, version):
        assert isinstance(version, str), """ """
        if version.count('.') == 2:
            self._major, self._minor, self._patch = version.split('.')
        elif version.count('-') == 2:
            self._major, self._minor, self._patch = version.split('-')
        else:
            raise AssertionError()

    @hybridmethod
    def _dump(cls) -> dict:
        __cslot__ = filter(lambda v: v != '_dataset', cls.__slot__)
        return dict(map(lambda v: (f'_{v}__', getattr(cls, v)), __cslot__))

    @_dump.instancemethod
    def _dump(self) -> dict:
        return dict(map(lambda v: (f'_{v}__', getattr(self, v)), self.__slot__))

    @hybridmethod
    def _load(cls, _dict):
        pass # update self._features

    @_load.instancemethod
    def _load(self, _dict):
        pass

    def __eq__(self, other):
        assert isinstance(other, self.__class__)
        return all(s_version == o_version 
                for s_version, o_version in 
            zip(self.__slot__, other.__slot__))

    def __hash__(self):
        return None

    def __repr__(self):
        return f'{self.__class__.__name__}(version={self.version})'

@dataclass
class Metadata:
    # Set in the dataset scripts
    description: str = field(default_factory=str)
    citation: str = field(default_factory=str)
    homepage: str = field(default_factory=str)
    license: str = field(default_factory=str)
    features: Features = field(default_factory=Features)
    labels: Features = field(default_factory=Features)
    #post_processed: Optional[PostProcessedInfo] = None
    #supervised_keys: Optional[SupervisedKeysData] = None
    #task_templates: Optional[List[TaskTemplate]] = None

    # Set later by the builder
    builder_name: Optional[str] = None
    config_name: Optional[str] = None
    version: Optional[Union[str, Version]] = None
    
    # Set later by `download_and_prepare`
    splits: Optional[dict] = None
    download_checksums: Optional[dict] = None
    download_size: Optional[int] = None
    post_processing_size: Optional[int] = None
    dataset_size: Optional[int] = None
    size_in_bytes: Optional[int] = None



#Dataset.MetaData.Info
#Dataset.MetaData.Version
#Dataset.MetaData.Checksum
#Dataset.Builder.Value
#Dataset.Builder
#Dataset.Plotter
