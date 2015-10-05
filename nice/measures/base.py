from ..externals.h5io import write_hdf5, read_hdf5
from ..utils import write_hdf5_mne_epochs
import numpy as np

from mne.utils import logger
from mne.epochs import _compare_epochs_infos
from mne.io.meas_info import Info
import h5py


class BaseMeasure(object):
    """Base class for M/EEG measures"""

    def _save_info(self, fname):
        has_ch_info = False
        with h5py.File(fname) as h5fid:
            if 'nice/data/ch_info' in h5fid:
                has_ch_info = True
                logger.info('Channel info already present in HDF5 file, '
                            'will not be overwritten')

        if not has_ch_info:
            ch_info = self.ch_info_
            logger.info('Writing channel info to HDF5 file')
            write_hdf5(fname, ch_info, title='nice/data/ch_info')

    def _get_save_vars(self, exclude):
        return {k: v for k, v in vars(self).items() if
                k not in exclude}

    def save(self, fname):
        self._save_info(fname)
        save_vars = self._get_save_vars(exclude=['ch_info_'])
        write_hdf5(
            fname,
            save_vars,
            title=_get_title(self.__class__, self.comment))

    def fit(self, epochs):
        self.ch_info_ = epochs.info
        self._fit(epochs)
        return self

    def transform(self, epochs):
        self._transform(epochs)
        return self

    def _get_title(self):
        return _get_title(self.__class__, self.comment)

    def _reduce_to(self, reduction_func, target, picks):
        if not hasattr(self, 'data_'):
            raise ValueError('You did not fit me. Do it again after fitting '
                             'some data!')
        out = self.get_picked_data(picks)
        for axis, func in self._expand_reduction(reduction_func, target):
            out = func(out, axis=axis)
        return out

    def reduce_to_topo(self, reduction_func, picks=None):
        return self._reduce_to(
            reduction_func, target='topography', picks=picks)

    def reduce_to_scalar(self, reduction_func, picks=None):
        return self._reduce_to(reduction_func, target='scalar', picks=picks)

    @classmethod
    def _read(cls, fname, comment='default'):
        return _read_measure(cls, fname, comment=comment)


class BaseEventRelated(BaseMeasure):

    def fit(self, epochs):
        self.ch_info_ = epochs.info
        self.shape_ = epochs.get_data().shape
        self.epochs_ = epochs
        self.data_ = epochs.get_data()
        return self

    def save(self, fname):
        self._save_info(fname)
        save_vars = self._get_save_vars(
            exclude=['ch_info_', 'data_', 'epochs_'])

        has_epochs = False
        with h5py.File(fname) as h5fid:
            if 'nice/data/epochs' in h5fid:
                has_epochs = True
                logger.info('Epochs already present in HDF5 file, '
                            'will not be overwritten')

        if not has_epochs:
            epochs = self.epochs_
            logger.info('Writing epochs to HDF5 file')
            write_hdf5_mne_epochs(fname, epochs)
        write_hdf5(
            fname, save_vars,
            title=_get_title(self.__class__, self.comment))

    @classmethod
    def _read(cls, fname, epochs, comment='default'):
        return _read_event_related(cls, fname=fname, epochs=epochs,
                                   comment=comment)

    def _get_title(self):
        return _get_title(self.__class__, self.comment)


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
    attrs['ch_info_'] = Info(read_hdf5(fname, title='nice/data/ch_info'))
    out = klass(**init_params)
    for k, v in attrs.items():
        if k.endswith('_'):
            setattr(out, k, v)
    return out


def _check_epochs_consistency(info1, info2, shape1, shape2):
    _compare_epochs_infos(info1, info2, 2)
    np.testing.assert_array_equal(shape1, shape2)


def _read_event_related(cls, fname, epochs, comment='default'):
    out = _read_measure(cls, fname, comment=comment)
    shape1 = epochs.get_data().shape
    shape2 = out.shape_
    _check_epochs_consistency(out.ch_info_, epochs.info, shape1, shape2)
    out.epochs_ = epochs
    out.data_ = epochs.get_data()
    return out
