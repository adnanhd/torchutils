import torch
import numpy
import typing
import torchvision
from .transforms import NumpyToTensor


class NumpyDataset(torch.utils.data.dataset.Dataset):
    __slots__ = ('inputs', 'input_transforms',
                 'labels', 'label_transforms')

    def __init__(self, inputs, labels, input_transforms=[], label_transforms=[]):
        super().__init__()

        if len(inputs) != len(labels):
            raise AssertionError('features and labels expected to be same size but '
                                 f'found {inputs.__len__()} != {labels.__len__()}')
        
        if isinstance(inputs, numpy.ndarray):
            input_transforms.insert(0, NumpyToTensor())
        if isinstance(labels, numpy.ndarray):
            label_transforms.append(0, NumpyToTensor())

        self.inputs = inputs
        self.labels = labels
        self.input_transforms = torchvision.transforms.Compose(input_transforms)
        self.label_transforms = torchvision.transforms.Compose(label_transforms)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> typing.Tuple[numpy.ndarray]:
        indexed_input = self.input_transforms(self.inputs[index])
        indexed_label = self.label_transforms(self.labels[index])
        return indexed_input, indexed_label

    # LOAD SAVE methodlari, split vs methodlari