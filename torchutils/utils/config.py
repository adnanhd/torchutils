import yaml
import json
import os.path as osp
from pydantic import BaseModel
from configparser import ConfigParser


class Config(BaseModel):
    def to_cfg(self, path: str):
        parser = ConfigParser()
        parser.add_section(self.__class__.__qualname__)
        for name, value in self.dict().items():
            parser.set(self.__class__.__qualname__, name, str(value))
        with open(path, 'w') as f:
            parser.write(f)

    @classmethod
    def from_cfg(cls, path: str):
        if not osp.isfile(path):
            raise FileNotFoundError(
                f'path {path} is not found by {cls.__qualname__}'
            )
        parser = ConfigParser()
        parser.read(path)
        # section not found error
        if not parser.has_section(cls.__qualname__):
            raise KeyError(
                f'section {cls.__qualname__} is not found in file {path}'
            )
        return cls(**dict(parser.items(cls.__qualname__)))

    def to_yaml(self, path: str):
        if not osp.isfile(path):
            data = dict()
        else:
            data = None
        with open(path, 'w') as file:
            if data is None:
                data = yaml.safe_load(file)
            data[self.__class__.__qualname__] = self.dict()
            yaml.safe_dump(data, file)

    @classmethod
    def from_yaml(cls, path: str):
        if not osp.isfile(path):
            raise FileNotFoundError(
                f'path {path} is not found by {cls.__qualname__}'
            )
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        if cls.__qualname__ not in data.keys():
            raise KeyError(
                f'key {cls.__qualname__} is not found in file {path}'
            )
        return cls(**data[cls.__qualname__])

    def to_json(self, path: str):
        if not osp.isfile(path):
            data = dict()
        else:
            data = None
        with open(path, 'w') as file:
            if data is None:
                data = json.load(file)
            data[self.__class__.__qualname__] = self.dict()
            json.dump(data, fp=file)

    @classmethod
    def from_json(cls, path: str):
        if not osp.isfile(path):
            raise FileNotFoundError(
                f'path {path} is not found by {cls.__qualname__}'
            )
        with open(path, 'r') as file:
            data = json.load(file)
        if cls.__qualname__ not in data.keys():
            raise KeyError(
                f'key {cls.__qualname__} is not found in file {path}'
            )
        return cls(**data[cls.__qualname__])
