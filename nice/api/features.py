from mne.utils import logger

from .modules import _get_module_func


def fit(instance, config='default', config_params=None):
    logger.info('Processing features from {} config'.format(config))
    out = None
    if config_params is None:
        config_params = {}
    func = _get_module_func('features', config)
    out = func(config_params=config_params)
    out.fit(instance)
    return out
