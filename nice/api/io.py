import os.path as op
from glob import glob

from mne.utils import logger

from .modules import _get_module_func

_io_config_suffix_map = {
    'default': 'default-epochs.fif',
}


def register_suffix(config, suffix):
    if config in _io_config_suffix_map:
        logger.warning('Overwriting IO suffix for {}'.format(config))
    _io_config_suffix_map[config] = suffix


def _check_io_suffix(path, config, multiple):
    suffix = _io_config_suffix_map[config]
    files = glob(op.join(path, '*{}'.format(suffix)))
    if multiple is False and len(files) != 1:
        msg = 'Only one file must have the {} suffix: {}'.format(config, suffix)
        logger.error(msg)
        raise ValueError(msg)
    elif multiple is True and len(files) == 0:
        msg = ('At least one file must have the {} '
               'suffix: {}'.format(config, suffix))
        logger.error(msg)
        raise ValueError(msg)
    return files


def read(path, config='default', config_params=None):
    logger.info('Reading data using {} config'.format(config))
    if config_params is None:
        config_params = {}
    out = None
    func = _get_module_func('io', config)
    out = func(path, config_params=config_params)
    return out
