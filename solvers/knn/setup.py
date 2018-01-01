from distutils import core
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

ext_modules = [Extension("src.solvers.cov.rating_cov",
                         ["src/solvers/cov/rating_cov.py"],
                         include_dirs=[numpy.get_include()])]

core.setup(
    name='Netflix Prize - rating_cov',
    ext_modules=cythonize(ext_modules)
)
