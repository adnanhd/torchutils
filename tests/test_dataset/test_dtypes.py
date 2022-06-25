from torchutils.data.dtypes import DType, NpTorchType
import torch
import numpy as np

ignore_NpTorchDtype=True
myVarTorch = torch.empty(3, 5)
myVarNumPy = np.empty((3, 5))


def test_NpTorchDtype_np(): assert ignore_NpTorchDtype or isinstance(np.float32, NpTorchType)

def test_NpTorchDtype_torch(): assert ignore_NpTorchDtype or isinstance(torch.float, NpTorchType)

def test_DType(): 
    myType = DType(np.float32, shape=(12,))
    myVarTorch = np.random.rand(12).reshape((12,)).astype('float32')
    assert isinstance(myVarTorch, myType)

def test_string(): assert isinstance('1', DType('string'))
def test_integer(): assert isinstance(1, DType('integer'))
def test_boolean(): assert isinstance(True, DType('boolean'))
def test_float(): assert isinstance(1.0, DType('float'))
def test_torch_float16(): assert isinstance(myVarTorch.to(dtype=torch.float16), DType('torch.float16', shape=(3, 5)))
def test_torch_float32(): assert isinstance(myVarTorch.to(dtype=torch.float32), DType('torch.float32', shape=(3, 5)))
def test_torch_float64(): assert isinstance(myVarTorch.to(dtype=torch.float64), DType('torch.float64', shape=(3, 5)))
def test_torch_int8(): assert isinstance(myVarTorch.to(dtype=torch.int8), DType('torch.int8', shape=(3, 5)))
def test_torch_int16(): assert isinstance(myVarTorch.to(dtype=torch.int16), DType('torch.int16', shape=(3, 5)))
def test_torch_int32(): assert isinstance(myVarTorch.to(dtype=torch.int32), DType('torch.int32', shape=(3, 5)))
def test_torch_int64(): assert isinstance(myVarTorch.to(dtype=torch.int64), DType('torch.int64', shape=(3, 5)))
def test_torch_uint8(): assert isinstance(myVarTorch.to(dtype=torch.uint8), DType('torch.uint8', shape=(3, 5)))
def test_np_float16(): assert isinstance(np.float16(myVarNumPy), DType('np.float16', shape=(3, 5)))
def test_np_float32(): assert isinstance(np.float32(myVarNumPy), DType('np.float32', shape=(3, 5)))
def test_np_float64(): assert isinstance(np.float64(myVarNumPy), DType('np.float64', shape=(3, 5)))
def test_np_float128(): assert isinstance(np.float128(myVarNumPy), DType('np.float128', shape=(3, 5)))
def test_np_int8(): assert isinstance(np.int8(myVarNumPy), DType('np.int8', shape=(3, 5)))
def test_np_int16(): assert isinstance(np.int16(myVarNumPy), DType('np.int16', shape=(3, 5)))
def test_np_int32(): assert isinstance(np.int32(myVarNumPy), DType('np.int32', shape=(3, 5)))
def test_np_int64(): assert isinstance(np.int64(myVarNumPy), DType('np.int64', shape=(3, 5)))
def test_np_uint8(): assert isinstance(np.uint8(myVarNumPy), DType('np.uint8', shape=(3, 5)))
def test_np_uint16(): assert isinstance(np.uint16(myVarNumPy), DType('np.uint16', shape=(3, 5)))
def test_np_uint32(): assert isinstance(np.uint32(myVarNumPy), DType('np.uint32', shape=(3, 5)))
def test_np_uint64(): assert isinstance(np.uint64(myVarNumPy), DType('np.uint64', shape=(3, 5)))
