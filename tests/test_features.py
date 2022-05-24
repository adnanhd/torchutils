import numpy as np
#from torchutils.data.features import Features
from torchutils.data.dtypes import DType
from pydantic import BaseModel


class Features(BaseModel):
    __prepare__ = {}

    @classmethod
    def prepare(cls, *keys):
        import pdb; pdb.set_trace()
        #print(cls.__fields__.keys(), cls)
        #assert all(key in cls.__fields__ for key in keys)
        def prepare_decorator(fn):
            #cls.__prepare__[keys] = fn
            import pdb; pdb.set_trace()
            return fn
        return prepare_decorator


class TestFeatures(Features):
    x: DType(np.float32, shape=(12,))
    y: DType(np.float32, shape=(1,))

    @Features.prepare('x', 'y')
    def prepare_io(self, path):
        pass

