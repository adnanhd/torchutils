import os

from .utils import Version

from dataclasses import dataclass, field
from typing import Optional, Union
from ..utils import Config


@dataclass
class Builder(Config):
    _DATASET = None
    """ A dataset preprocessor.
    @brief Converts a metadata to a dataset by downloading and
    preprocessing etc. data
    """
    # Set later by the builder
    dataset_path: str = field(default=None)
    builder_name: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    version: Optional[Union[str, Version]] = field(default=None)

    @dataclass
    class Config(Config):
        # Set later by `download_and_prepare`
        splits: Optional[dict] = field(default=None)
        download_checksums: Optional[dict] = field(default=None)
        download_size: Optional[int] = field(default=None)
        post_processing_size: Optional[int] = field(default=None)
        dataset_size: Optional[int] = field(default=None)
        size_in_bytes: Optional[int] = field(default=None)

        def __post_init__(self):
            super().__init__('download')

    _config: Optional[Config] = field(default_factory=Config)

    def __post_init__(self):
        assert os.path.isdir(self.dataset_path), "Path must be valid"
        # super().__init__('builder')
        self._dlist: Optional[list] = None

    @property
    def config(self):
        return self._config

    def listdir(self, path: str):
        return list(self._listdir(path))

    def __len__(self):
        return len(self._dlist)

    def _listdir(self, path: str):
        raise NotImplementedError()

    def prepare(self):
        raise NotImplementedError()

    def download(self):
        raise NotImplementedError()

    def _generate(self, path: str):  # generates one example at a call
        raise NotImplementedError()

    def __iter__(self):
        dirname = self.dataset_path
        self._dlist = self.listdir(dirname)
        return self

    def __next__(self):
        if len(self._dlist):
            basename = self._dlist.pop(0)
            dirname = self.dataset_path
            return self._generate(os.path.join(dirname, basename))
        else:
            self._dlist = None
            raise StopIteration
