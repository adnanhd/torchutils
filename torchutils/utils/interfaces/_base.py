from pydantic.dataclasses import dataclass
from pydantic import RootModel
from abc import ABC
import yaml
import json
import os


class RegisterationError(ValueError):
    pass


class _BaseModelType(ABC):
    @classmethod
    def __get_validators__(cls):
        yield cls.field_validator

    @classmethod
    def field_validator(cls, field_type, info):
        if isinstance(field_type, cls):
            return field_type
        raise ValueError(f"{field_type} is not a {cls}")


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