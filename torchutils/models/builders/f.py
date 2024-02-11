import torch
import inspect
from ...utils import BuilderType, FunctionalType
from ._dev_utils import obtain_registered_kwargs


class Functional(FunctionalType):
    @classmethod
    def field_signature_validator(cls, field_type, info):
        signature = inspect.signature(field_type)
        assert signature.parameters['input'].annotation == torch.Tensor
        assert signature.parameters['target'].annotation == torch.Tensor
        return field_type


class Criterion(BuilderType):
    @classmethod
    def builder_from_class(cls, field_type, info):
        if inspect.isclass(field_type) and cls.__subclasscheck__(field_type):
            config = obtain_registered_kwargs(field_type, info.data['arguments'])
            device = info.data['arguments'].get('device', None)
            return field_type(**config).to(device=device)
        return field_type


Criterion.register_subclasses(torch.nn.modules.loss._Loss)


for loss_func in vars(torch.functional.F).values():
    try:
        Functional.register(loss_func)
    except ValueError:
        pass
    except KeyError: 
        pass
    except AssertionError:
        pass
