from setuptools import setup

__author__ = 'Adnan Harun Dogan'
__email__ = 'adnanharundogan@gmail.com'

with open('README.md') as f:
    long_description = f.readlines()

with open('requirements.txt') as f:
    required_packages = f.readlines()

setup(
    name='torchutils',
    version='1.3.2',
    description=long_description,
    author=__author__,
    author_email=__email__,
    install_requires=required_packages,
)
