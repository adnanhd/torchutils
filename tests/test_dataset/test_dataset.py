from torchutils.data import Dataset
import numpy as np

def test_dataset():
    data = Dataset(
    features=np.zeros((44, 256, 256)).astype('f4'),
    labels=np.zeros((44, 256, 256)).astype('f4')
        )

    assert data
