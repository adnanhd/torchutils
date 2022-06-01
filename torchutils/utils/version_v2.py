from dataclasses import dataclass, field
from typing import Union

@dataclass
class Version(object):
    major: int = field() 
    minor: int = field() 
    patch: Union[int, str] = field()

    def __lt__(self, other: "Version") -> bool:
        assert isinstance(other, self.__class__)
        if self.major < other.major: return True
        elif self.major > other.major: return False
        if self.minor < other.minor: return True
        elif self.minor > other.minor: return False
        if self.patch < other.patch: return True
        elif self.patch > other.patch: return False
        return False

    def __gt__(self, other: "Version") -> bool:
        assert isinstance(other, self.__class__)
        if self.major > other.major: return True
        elif self.major < other.major: return False
        if self.minor > other.minor: return True
        elif self.minor < other.minor: return False
        if self.patch > other.patch: return True
        elif self.patch < other.patch: return False
        return False

    def __eq__(self, other: "Version") -> bool:
        assert isinstance(other, self.__class__)
        if self.major != other.major: return False
        if self.minor != other.minor: return False
        if self.patch != other.patch: return False
        return True
    
    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.major}.{self.minor}.{self.patch})"

