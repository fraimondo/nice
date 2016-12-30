from . io import read
from . preprocessing import preprocess
from . features import fit
from . reductions import get_reductions
from . import modules
from . default import register as _register_default

_register_default()
try:
    import next
    print('Using NICE Extensions')
except ImportError:
    print('NICE Extensions not installed')
