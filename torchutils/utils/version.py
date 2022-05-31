import os
import math
import copy
import torch
import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .mthdutils import hybridmethod
    


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
    _major: Union[int, str] = field(default=0)
    _minor: Union[int, str] = field(default=0)
    _patch: Union[int, str] = field(default=0)
    def __init__(self, major=0, minor=0, patch=0, version=None):
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

    @classmethod
    def fromstring(cls, string):
        return cls(string)

    @property
    def version(self):
        return f'{self._major}.{self._minor}.{self._patch}'

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
    def to_dict(cls) -> dict:
        __cslot__ = filter(lambda v: v != '_dataset', cls.__slot__)
        return dict(map(lambda v: (f'_{v}__', getattr(cls, v)), __cslot__))

    @to_dict.instancemethod
    def to_dict(self) -> dict:
        return dict(map(lambda v: (f'_{v}__', getattr(self, v)), self.__slot__))

    @hybridmethod
    def from_dict(cls, _dict):
        pass # update self._features

    @from_dict.instancemethod
        pass

        assert isinstance(other, self.__class__)
    def __eq__(self, other):
        return all(s_version == o_version 
                for s_version, o_version in 
            zip(self.__slot__, other.__slot__))

    def __hash__(self):
        return None

    def __repr__(self):
        return f'{self.__class__.__name__}(version={self.version})'

