from distutils import core
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

ext_modules = [Extension("rating_cov",
                         ["rating_cov.pyx"],
                         include_dirs=[numpy.get_include()],
                         language="c++",
                         ),
               Extension("predict_knn",
                         ["predict_knn.pyx"],
                         include_dirs=[numpy.get_include()],
                         #language="c++",
                         )
               ]

core.setup(
    name='Netflix Prize',
    ext_modules=cythonize(ext_modules),
)
