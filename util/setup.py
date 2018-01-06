from distutils import core
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

ext_modules = [Extension("user_cor",
                         ["user_cor.pyx"],
                         include_dirs=[numpy.get_include()],
                         language="c++",
                         ),
               Extension("mle_cov",
                         ["mle_cov.pyx"],
                         include_dirs=[numpy.get_include()],
                         language="c++",
                         ),
               Extension("incomplete_projection",
                         ["incomplete_projection.pyx"],
                         include_dirs=[numpy.get_include()],
                         ),
               ]

core.setup(
    name='Netflix Prize',
    ext_modules=cythonize(ext_modules),
)
