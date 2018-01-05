from distutils import core
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

ext_modules = [Extension("rating_cor",
                         ["rating_cor.pyx"],
                         include_dirs=[numpy.get_include()],
                         language="c++",
                         )]

core.setup(
    name='Netflix Prize',
    ext_modules=cythonize(ext_modules),
)
