from distutils.core import setup, Extension

from numpy import get_include

include = [get_include(), '.']

module1 = Extension('ompk',
                    sources=['ompk.c', 'komplexity.c'],
                    extra_compile_args=["-fopenmp", "-std=c99", "-O3"],
                    extra_link_args=['-lz'],
                    include_dirs=include)

setup(name='OmpkBindings',
      version='1.0',
      description='This is the OmpK bindings to python interface',
      ext_modules=[module1])
