from typing import Union
from pydantic import BaseModel


class Version(BaseModel):
    major: int
    minor: int
    patch: Union[int, str]

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], str):
            major, minor, patch = args[0].split('.')
            super(Version, self).__init__(
                major=int(major), minor=int(minor), patch=patch)
        else:
            major, minor, patch = args
            super(Version, self).__init__(
                major=major, minor=minor, patch=patch)

    def __lt__(self, other: "Version") -> bool:
        assert isinstance(other, self.__class__)
        if self.major < other.major:
            return True
        elif self.major > other.major:
            return False
        if self.minor < other.minor:
            return True
        elif self.minor > other.minor:
            return False
        if self.patch < other.patch:
            return True
        elif self.patch > other.patch:
            return False
        return False

    def __gt__(self, other: "Version") -> bool:
        assert isinstance(other, self.__class__)
        if self.major > other.major:
            return True
        elif self.major < other.major:
            return False
        if self.minor > other.minor:
            return True
        elif self.minor < other.minor:
            return False
        if self.patch > other.patch:
            return True
        elif self.patch < other.patch:
            return False
        return False

    def __eq__(self, other: "Version") -> bool:
        assert isinstance(other, self.__class__)
        if self.major != other.major:
            return False
        if self.minor != other.minor:
            return False
        if self.patch != other.patch:
            return False
        return True

    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.major}" \
               f".{self.minor}.{self.patch})"
