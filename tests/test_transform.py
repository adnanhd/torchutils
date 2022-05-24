import numpy as np
from transform import MinMaxTransform

data = np.ones((10, 10))

#key = MinMaxTransform(data, axis=(1,))
data[2,3] = 0.0
data[1,3] = 2.0


with MinMaxTransform(data, axis=None) as key:
    print(data)


def test_obtain_shape_1():
    f = np.random.rand(204, 4, 256, 256)
    shape = transform._obtain_shape((0, 2, 3), f.shape)
    assert shape == (1, 4, 1, 1)

def test_obtain_shape_2():
    f = np.random.rand(204, 4, 256, 256)
    shape = transform._obtain_shape((-1, -3), f.shape)
    assert shape == (204, 1, 256, 1)

def test_obtain_shape_3():
    f = np.random.rand(204, 4, 256, 256)
    shape = transform._obtain_shape(None, f.shape)
    assert shape == (1, 1, 1, 1)
