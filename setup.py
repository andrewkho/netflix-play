from distutils import core
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

ext_modules = [Extension("solvers.svd_neighbour.svd_neighbour_predict",
                         ["solvers/svd_neighbour/svd_neighbour_predict.pyx"],
                         include_dirs=[numpy.get_include()],
                         ),
               Extension("util.user_cor",
                         ["util/user_cor.pyx"],
                         include_dirs=[numpy.get_include()],
                         language="c++",
                         ),
               Extension("util.mle_cov",
                         ["util/mle_cov.pyx"],
                         include_dirs=[numpy.get_include()],
                         ),
               Extension("util.incomplete_projection",
                         ["util/incomplete_projection.pyx"],
                         include_dirs=[numpy.get_include()],
                         ),
               ]

core.setup(
    name='Netflix Prize',
    ext_modules=cythonize(ext_modules),
)
