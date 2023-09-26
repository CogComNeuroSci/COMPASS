from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup (
    name = 'wfpt_n',
    ext_modules = cythonize("wfpt_n.pyx"),
    include_dirs=[np.get_include()]
)