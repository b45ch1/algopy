
# compile with
# python setup.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("vector_forward_mode", ["vector_forward_mode.pyx"])]
)
 
