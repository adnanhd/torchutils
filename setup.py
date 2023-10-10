from setuptools import setup

setup(
    name=cfg.get('metadata', 'name'),
    version='2.0.0',
    description=open('README.md').read(),
    author='adnanharundogan',
    author_email='adnanharundogan@gmail.com',
    license='MIT',
    install_requires=open('requirements.txt').read().strip(),
)
