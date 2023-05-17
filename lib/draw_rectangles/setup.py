from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(name="draw_rectangles_cython", ext_modules=cythonize('draw_rectangles.pyx'), include_dirs=[numpy.get_include()])