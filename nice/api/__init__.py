from . io import read
from . preprocessing import preprocess
from . features import fit
from . reductions import get_reductions
from . report import create_report
from . summarize import summarize_subject
from . import modules
from . default import register as _register_default

import pip
_register_default()

extensions = [x for x in map(lambda x: x.key, pip.get_installed_distributions())
              if x.startswith('next')]
for x in extensions:
    try:
        mod = __import__(x.replace('-', '_'))
        print('Using {}'.format(mod.__next_name__))
    except ImportError:
        print('Error loading {}'.format(x))
