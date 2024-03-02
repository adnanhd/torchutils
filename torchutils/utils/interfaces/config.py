from pydantic.dataclasses import dataclass
from pydantic import RootModel
import os
import yaml
import json
import typing
import abc
import inspect


class _BaseConfig:
    def to_dict(self):
        return RootModel[self.__class__](self).model_dump()

    def to_json(self, **kwds):
        return json.dumps(self.to_dict(), **kwds)

    def to_yaml(self, **kwds):
        return yaml.dump(self.to_dict(), **kwds)

    @classmethod
    def from_json(cls, config_fname):
        assert os.path.isfile(config_fname)

        with open(config_fname, "r") as f:
            try:
                return cls(**json.load(f))
            except json.JSONDecodeError as exc:
                print(f"Error reading JSON file: {exc}")

    @classmethod
    def from_yaml(cls, config_fname):
        assert os.path.isfile(config_fname)

        with open(config_fname, "r") as f:
            try:
                return cls(**yaml.safe_load(f))
            except yaml.YAMLError as exc:
                print(f"Error reading YAML file: {exc}")


@dataclass
class BaseConfig(_BaseConfig):
    pass


class _BaseConfigType(abc.ABC):

    @classmethod
    def get_dataclass(cls, class_name: str):
        """Retrieves the registered instance by its class name."""
        fn_sign = inspect.signature(cls.get_functional(class_name)).parameters

        def maybe_annotate(subcls):
            return subcls if subcls != inspect._empty else typing.Any

        config_class = type(
                class_name + '_DataClass',
                (_BaseConfig,),
                {
                    **{k: v.default for k, v in fn_sign.items()
                       if v.default != inspect._empty},

                    '__annotations__': {k: maybe_annotate(v.annotation)
                                        for k, v in fn_sign.items()}
                }
            )

        return dataclass(config_class)