import typing
import inspect


class ArgumentError(Exception):
    pass


def determine_types(args, kwargs):
    return tuple(type(a) for a in args), \
        tuple((k, type(v)) for k, v in kwargs.items())


def obtain_registered_kwargs(fn: typing.Callable,
                             kwargs: typing.Dict[str, typing.Any]):
    parameters = inspect.signature(fn).parameters.keys()
    return dict(
        filter(
            lambda item: item[0] in parameters,
            kwargs.items()
        )
    )
