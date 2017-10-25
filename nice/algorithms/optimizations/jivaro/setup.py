from distutils.core import setup, Extension

from numpy import get_include

include = [get_include(), '.']

module1 = Extension('jivaro',
                    sources=['jivaro.c', 'symb_transf.c', 'blocktrie.c', 'smi.c'],
                    extra_compile_args=['-fopenmp', '-std=c99', '-O3'],
                    extra_link_args=['-fopenmp'],
                    include_dirs=include)

setup(name='JivaroBindings',
      version='1.0',
      description='This is the Jivaro Proyect bindings to python interface',
      ext_modules=[module1])
