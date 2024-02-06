from setuptools import setup
from setuptools import Extension
import numpy


extension = Extension(
    'conv_func',
    ['conv_func.c'],
    extra_compile_args=['/O2']
)
setup(
    name='conv-lib',
    version='2.1',
    description='C extension made specifically for the convolution operations in the neural network project: https://github.com/GeyaWang/py-nn.git',
    install_requires=['numpy'],
    ext_modules=[extension],
    include_dirs=[numpy.get_include()]
)
