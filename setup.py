from setuptools import setup
from configparser import ConfigParser
from torchutils import __version__

cfg = ConfigParser()
cfg.read('setup.cfg')

__author__ = cfg.get('metadata', 'maintainer_name')
__email__ = cfg.get('metadata', 'maintainer_email')
__url__ = cfg.get('metadata', 'url')

with open('README.md') as f:
    long_description = "\n".join(f.readlines())

with open('requirements.txt') as f:
    required_packages = f.readlines()

setup(
    name=cfg.get('metadata', 'name'),
    version=str(__version__),
    description=long_description,
    author=__author__,
    author_email=__email__,
    install_requires=required_packages,
)
