from mne.utils import logger

from .modules import _get_module_func


def preprocess(instance, config='default', config_params=None):
    logger.info('Preprocessing from {} config'.format(config))
    out = None
    if config_params is None:
        config_params = {}
    func = _get_module_func('preprocess', config)
    out = func(instance, config_params=config_params)
    return out


def _check_min_events(epochs, bad_epochs, min_events):
    n_orig_epochs = len([x for x in epochs.drop_log if 'IGNORED' not in x])
    if isinstance(min_events, float):
        logger.info('Using relative min_events: {} * {} = {} '
                    'epochs remaining to reject preprocess'.format(
                        min_events, n_orig_epochs,
                        int(n_orig_epochs * min_events)))
        min_events = int(n_orig_epochs * min_events)

    epochs_remaining = len(epochs)
    if epochs_remaining < min_events:
        msg = ('Can not clean data. Only {} out of {} epochs '
               'remaining.'.format(epochs_remaining, n_orig_epochs))
        logger.error(msg)
        raise ValueError(msg)


def _check_min_channels(epochs, bad_channels, min_channels):
    if isinstance(min_channels, float):
        logger.info('Using relative min_channels: {} * {} = {} '
                    'channels remaining to reject preprocess'.format(
                        min_channels, epochs.info['nchan'],
                        epochs.info['nchan'] * min_channels))
        min_channels = int(epochs.info['nchan'] * min_channels)

    chans_remaining = epochs.info['nchan'] - len(bad_channels)
    if chans_remaining < min_channels:
        msg = ('Can not clean data. Only {} out of {} channels '
               'remaining.'.format(chans_remaining, epochs.info['nchan']))
        logger.error(msg)
        raise ValueError(msg)
