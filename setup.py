from setuptools import setup
from setuptools import Extension
import numpy


extension = Extension(
    'conv_func',
    ['conv_func.c']
)
setup(
    name='conv-lib',
    version='2.2',
    install_requires=['numpy'],
    ext_modules=[extension],
    include_dirs=[numpy.get_include()]
)
