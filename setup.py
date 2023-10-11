from setuptools import setup

setup(
    name='torchutils',
    version='2.0.1',
    description=open('README.md').read(),
    author='adnanharundogan',
    author_email='adnanharundogan@gmail.com',
    license='MIT',
    install_requires=open('requirements.txt').read().strip(),
)
