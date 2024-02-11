import torchutils.models.builders as b
import pydantic
import torch


@b.NeuralNet.register
class MyNetwork(torch.nn.Module):
    pass


class Foo(pydantic.BaseModel):
    arguments: dict
    model: b.NeuralNet


net = Foo(model='MyNetwork', arguments=dict())