from typing import Union, Optional
from dataclasses import dataclass, field
from ..utils import BaseConfig, Version


@dataclass
class Metadata():
    from .dtypes import DType
    from .features import Features

    # Set in the dataset scripts
    description: str = field(default_factory=str)
    citation: str = field(default_factory=str)
    homepage: str = field(default_factory=str)
    license: str = field(default_factory=str)
    features: Features = field(default_factory=Features)
    labels: Features = field(default_factory=Features)
    version: Optional[Union[str, Version]] = None
    #post_processed: Optional[PostProcessedInfo] = None
    #supervised_keys: Optional[SupervisedKeysData] = None
    #task_templates: Optional[List[TaskTemplate]] = None
    """
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
    """

    def __post_init__(self):
        if not isinstance(self.features, self.Features):
            self.features = self.Features.new(self.features)
        if not isinstance(self.labels, self.Features):
            self.labels = self.Features.new(self.labels)
        super().__init__('metadata')

    def _keys(self):
        def ignore(s): return s not in ('features', 'labels')
        keys = super()._keys()
        return filter(ignore, keys)
