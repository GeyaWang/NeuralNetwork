from setuptools import setup
from setuptools import Extension
import numpy


conv_ext = Extension(
    'conv_func',
    ['conv_func.c']
)

setup(
    name='nn-func-lib',
    version='2.7',
    install_requires=['numpy'],
    ext_modules=[conv_ext],
    include_dirs=[numpy.get_include()]
)
