import torch
import typing
import inspect
# for typesytems
import torchvision.models as m
from ..utils import BuilderType, FunctionalType
# for registration
import abc
from ..utils import RegisterationError
import torch.optim.lr_scheduler as lr_sched


__all__ = ['Functional', 'Criterion', 'NeuralNet', 'Optimizer', 'Scheduler']


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


class NeuralNet(BuilderType):
    @classmethod
    def __get_validators__(cls):
        yield cls.get_class_from_name
        yield cls.torchvision_builder_from_name
        yield cls.builder_from_class
        yield cls.field_validator
            
    @classmethod
    def builder_from_class(cls, field_type, info):
        if inspect.isclass(field_type) and cls.__subclasscheck__(field_type) \
            or inspect.isfunction(field_type):
            config = obtain_registered_kwargs(field_type, info.data['arguments'])
            device = info.data['arguments'].get('device', None)
            return field_type(**config).to(device=device)
        return field_type

    @classmethod
    def torchvision_builder_from_name(cls, field_type, info):
        if isinstance(field_type, str) and field_type in m.list_models():
            return m.get_model_builder(field_type)
        return field_type
    

class Optimizer(BuilderType):
    @classmethod
    def builder_from_class(cls, field_type, info):
        if inspect.isclass(field_type) and cls.__subclasscheck__(field_type):
            model = info.data['model']
            kwargs = obtain_registered_kwargs(field_type, info.data['arguments'])
            return field_type(model.parameters(), **kwargs)
        return field_type


class Scheduler(BuilderType):
    @classmethod
    def builder_from_class(cls, field_type, info):
        if inspect.isclass(field_type) and cls.__subclasscheck__(field_type):
            optimizer = info.data['optimizer']
            kwargs = obtain_registered_kwargs(field_type, info.data['arguments'])
            return field_type(optimizer, **kwargs)
        return field_type


# register criterion
Criterion.register_subclasses(torch.nn.modules.loss._Loss)

# register functional
for loss_func in vars(torch.functional.F).values():
    try:
        Functional.register(loss_func)
    except (RegisterationError, KeyError, AssertionError):
        pass

    
# register networks
abc.ABCMeta.register(NeuralNet, torch.nn.Module)

# register optimizer
Optimizer.register_subclasses(torch.optim.Optimizer)

# register scheduler
try:
    Scheduler.register_subclasses(lr_sched.LRScheduler)
except AttributeError:
    Scheduler.register_subclasses(lr_sched._LRScheduler)

if not Scheduler.has_subclass(lr_sched.ReduceLROnPlateau):
    Scheduler.register(lr_sched.ReduceLROnPlateau)


def obtain_registered_kwargs(fn: typing.Callable,
                             kwargs: typing.Dict[str, typing.Any]):
    parameters = inspect.signature(fn).parameters.keys()
    reg_keys = tuple(filter(parameters.__contains__, kwargs.keys()))
    return dict(zip(reg_keys, map(kwargs.__getitem__, reg_keys)))

