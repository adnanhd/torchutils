import os
import torchutils.data as data
import meshio
import numpy as np
import matplotlib.pyplot as plt
from ..dataset import Dataset
from ..builder import Builder
from ..utils import Version, verbose, profile

class VTKBuilder(Dataset.Builder):
#class VTKBuilder(Builder):
    _aoa = 0

    def _listdir(self, path):
        is_angle_of_attack = lambda p: p.endswith(f'_AOA={self._aoa}')
        return filter(is_angle_of_attack, os.listdir(path))

    def _generate(self, path):
        _, basename = os.path.split(path)
        assert f'{basename}.vtk' in os.listdir(path)
        data = meshio.read(f'{path}/{basename}.vtk')
        labels = {
            'P': data.point_data['P'].reshape(256, 256, 1).astype('float64'),
            'U': data.point_data['v'].reshape(256, 256, 3, 1)[:, :, 0, :].astype('float64'),
            'V': data.point_data['v'].reshape(256, 256, 3, 1)[:, :, 1, :].astype('float64'),
            'T': data.point_data['t'].reshape(256, 256, 1).astype('float64'),
        }
        return {'DistFunc': labels['P']}, labels


class VTK(Dataset):
#class VTK:
    Builder = VTKBuilder
    BUILDER_CLASS = VTKBuilder
    BUILDER_CONFIG_CLASS = VTKBuilder.Config
    
    def __init__(self, path: str):
        kwargs = self._fromiter(self.Builder(dataset_path=path))
        super().__init__(**kwargs)

    @classmethod
    def _metadata(cls):
        # Set in the dataset scripts
        return cls.Metadata(
            features = cls.Metadata.Features(
                DistFunc = cls.Metadata.Value(np.float32, (256, 256))
                ),
            labels = cls.Metadata.Features(
                P=cls.Metadata.Value(np.float32, (256, 256)), 
                U=cls.Metadata.Value(np.float32, (256, 256)), 
                V=cls.Metadata.Value(np.float32, (256, 256)),
                T=cls.Metadata.Value(np.float32, (256, 256))
                ),
            )

