from mne.utils import logger

from .modules import _get_module_func


def predict(features, summary, config='default', config_params=None):
    """ Predict one subject (features) agains a set of features (summary)

    Parameters
    ----------
    features : str or nice.Features
        Features to use. If str, will look for an HDF5 file inside the folder.
    summary : str or nice.api.summarize.Summary
        Summary to use. If str, will look for a summary inside the folder
    config : str
        configuration to use
    config_params : str
        Extra dictionary of parameters to pass to the config-specific function
    Returns
    -------
    out : prediction summary

    """
    logger.info('Predicting from {} config'.format(config))
    out = None
    if config_params is None:
        config_params = {}
    func = _get_module_func('predict', config)
    out = func(features, summary, config_params=config_params)
    return out
