from setuptools import setup
from setuptools import Extension
import numpy


setup(
    name='conv-lib',
    version='1',
    ext_modules=[
        Extension(
            'conv_func',
            ['conv_func.c'],
            include_dirs=[numpy.get_include()],
            extra_compile_args=['/O2']
        )
    ],
)
