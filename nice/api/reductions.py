import numpy as np
from scipy.stats import trim_mean

from mne.utils import logger

from .modules import _get_module_func


def trim_mean90(a, axis=0):
    return trim_mean(a, proportiontocut=.05, axis=axis)


def trim_mean80(a, axis=0):
    return trim_mean(a, proportiontocut=.1, axis=axis)


_stats_functions = {
    'trim_mean80': trim_mean80,
    'trim_mean90': trim_mean90,
    'std': np.std,
    'mean': np.mean
}


def get_avaialable_functions():
    return list(_stats_functions.keys())


def get_function_by_name(funname):
    if funname not in _stats_functions:
        raise ValueError('Function {} does not exist in stats'.format(funname))
    return _stats_functions[funname]


def get_reductions(config='default', config_params=None):
    logger.info('Using reductions from {} config'.format(config))
    configs = config.split('/')
    config_fun = configs[-1]
    func = _get_module_func('reductions', config)
    out = func(config_fun, config_params=config_params)
    return out
