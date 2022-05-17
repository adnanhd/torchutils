from configparser import ConfigParser
from .mthdutils import hybridmethod

def read(path):
    config = ConfigParser()
    config.read(path)
    return config

def write(config, path):
    with open(path, 'w') as f:
        config.write(f)

def set(config, section, keys, dataclass):
    if not config.has_section(section): config.add_section(section)
    for key in keys: config.set(section, key, str(getattr(dataclass, key))) 

def get(config, section, keys, dataclass=None):
    kwargs = dict() if dataclass is None else None
    if not config.has_section(section): return None
    for key in keys: 
        if dataclass is None:
            kwargs[key] = config.get(section, key)
        else:
            setattr(dataclass, key, config.get(section, key))
    return kwargs

class INIObject(object):
    def __init__(self, section: str, public_only: bool =True):
        self._section = section
        self._public_only = public_only
    
    def keys(self):
        public_fields = lambda s: not s.startswith('_')
        keys = self._keys()
        if self._public_only: keys = filter(public_fields, keys)
        return tuple(keys)

    def _keys(self):
        if hasattr(self, '__slots__'): return self.__slots__
        else: return self.__dataclass_fields__

    def load(self, path: str):
        """ save into ini file """
        #get(read(path), self._section, self.keys(), self)
        config = read(path)
        if not config.has_section(self._section): 
            raise KeyError(f'{path} has no section {self._section}')
        for key in self.keys(): 
            setattr(self, key, config.get(self._section, key))

    def save(self, path: str): 
        """ load from ini file """
        #set(read(path), self._section, self.keys(), self)
        config = read(path)
        if not config.has_section(self._section): 
            config.add_section(self._section)
        for key in self.keys(): 
            config.set(self._section, key, str(getattr(self, key))) 
        write(config, path)

