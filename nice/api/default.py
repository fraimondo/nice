from mne.epochs import BaseEpochs, read_epochs
from mne.utils import logger
from . modules import register_module


def register():
    register_module('io', 'default', _read_default)
    register_module('preprocess', 'default', _preprocess_bypass)


def _read_default(path):
    from .io import _check_io_suffix
    config = 'icm/rs/raw/egi'
    files = _check_io_suffix(path, config, multiple=False)
    return read_epochs(files[0], preload=True)


def _preprocess_bypass(instance, config_params):
    if not isinstance(instance, BaseEpochs):
        msg = 'Default preprocessing is only defined for epochs'
        logger.error(msg)
        raise ValueError(msg)
    return instance
