from ..externals.h5io import write_hdf5, read_hdf5

import numpy as np

from mne.utils import logger
from mne.epochs import _compare_epochs_infos
from mne.io.meas_info import Info
import h5py


class BaseMeasure(object):
    """Base class for M/EEG measures"""

    def save(self, fname):
        write_hdf5(
            fname,
            vars(self),
            title=_get_title(self.__class__, self.comment))

    def fit(self, epochs):
        pass

    def transform(self):
        pass


class BaseEventRelated(BaseMeasure):

    def fit(self, epochs):
        self.epochs_info_ = epochs.info
        self.shape_ = epochs.get_data().shape
        self.epochs_ = epochs
        self.data_ = epochs.get_data()

    def save(self, fname):
        save_vars = {k: v for k, v in vars(self).items() if
                     k not in ['data_', 'epochs_']}
        has_epochs = False
        with h5py.File(fname) as h5fid:
            if 'nice/data/epochs' in h5fid:
                has_epochs = True
                logger.info('Epochs already present in HDF5 file, '
                            'will not be overwritten')

        epochs = self.epochs_
        if not has_epochs:
            logger.info('Writing epochs to HDF5 file')
            epochs_vars = {k: v for k, v in vars(epochs).items() if not
                           (k.startswith('_') or k != 'data_')}
            write_hdf5(fname, epochs_vars,
                       title='nice/data/epochs')

        write_hdf5(
            fname, save_vars,
            title=_get_title(self.__class__, self.comment))


class BaseSpectral(BaseMeasure):
    pass


class BaseConnectivity(BaseMeasure):
    pass


def _get_title(klass, comment):
    if 'measure' in klass.__module__:
        kind = 'measure'
    else:
        raise NotImplementedError('Oh no-- what is this?')

    return '/'.join([
        'nice', kind, klass.__name__, comment])


def _read_measure(klass, fname, comment='default'):
    data = read_hdf5(fname,  _get_title(klass, comment))
    init_params = {k: v for k, v in data.items() if not k.endswith('_')}
    attrs = {k: v for k, v in data.items() if k.endswith('_')}
    if 'epochs_info_' in attrs:
        attrs['epochs_info_'] = Info(attrs['epochs_info_'])
    out = klass(**init_params)
    for k, v in attrs.items():
        if k.endswith('_'):
            setattr(out, k, v)
    return out


def _check_epochs_consistency(info1, info2, shape1, shape2):
    _compare_epochs_infos(info1, info2, 2)
    np.testing.assert_array_equal(shape1, shape2)
