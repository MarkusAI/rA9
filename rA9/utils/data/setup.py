from distutils.core import  Extension,setup
from Cython.Build import cythonize

ext = Extension(name='dataloader',sources=["dataloader.pyx"])
setup(ext_modules=cythonize(ext))
