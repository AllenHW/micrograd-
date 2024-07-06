from setuptools import setup, find_packages

setup(
    name="npgrad",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=2.0.0',
    ],
)