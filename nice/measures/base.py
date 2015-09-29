from ..externals.h5io import write_hdf5, read_hdf5

import numpy as np

from mne.epochs import _compare_epochs_infos
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
        self.epochs_info_ = epochs.info1

    def save(self, fname):
        save_vars = vars(self)
        has_epochs = False
        with h5py.File(fname) as h5fid:
            if 'nice/data/epochs' not in h5fid:
                has_epochs = True

        if not has_epochs:
            write_hdf5(fname, vars(save_vars.pop('epochs_')),
                       title='nice/data/epochs')

        write_hdf5(
            fname,
            save_vars,
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
    data = read_hdf5(
        fname,  _get_title(klass, comment))
    out = klass(**{k: v for k, v in data.items() if not k.endswith('_')})
    for k, v in data.items():
        if k.endswith('_'):
            setattr(out, k, v)
    return out


def _check_epochs_consistency(epochs1, epochs2):
    _compare_epochs_infos(epochs1.info1, epochs2.info, 2)
    np.assert_equal(epochs1.get_data(), epochs2.get_data())
    return epochs2
