from setuptools import setup
from Cython.Build import cythonize

setup(
    name="load_functions",
    ext_modules=cythonize("load_functions.py", language_level="3"),
)
