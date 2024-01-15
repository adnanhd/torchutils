import typing


class RegisterationError(ValueError):
    pass


class ModelDictionary:
    __slots__ = ['__saved_models__']
    def __init__(self):
        self.__saved_models__ = dict()

    def register_model(self, name: typing.Optional[str] = None):
        def wrapper(subclass: Module) -> Module:
            key = name if name is not None else name.__name__
            if key in self.__saved_models__:
                raise RegisterationError("An entry is already registered "
                                         f"under the name '{key}'.")
            return self.__saved_models__.setdefault(name, subclass)
        return wrapper

    def get_model(self, name: str):
        try:
            return self.__saved_models__[name]
        except KeyError:
            raise RegisterationError(f"Unknown model {name}")
