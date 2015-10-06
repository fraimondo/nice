from collections import OrderedDict
from .utils import h5_listdir
from .externals.h5io import read_hdf5, write_hdf5
from .measures.base import BaseMeasure, BaseEventRelated
import sys
import inspect
import mne
from mne.io.meas_info import Info
import numpy as np


class Features(OrderedDict):

    def __init__(self, measures):
        OrderedDict.__init__(self)
        for meas in measures:
            self.add_measure(meas)
        if self._check_measures_fit():
            self.ch_info_ = self.values()[0].ch_info_

    def fit(self, epochs):
        for meas in self.values():
            meas.fit(epochs)
        self.ch_info_ = self.values()[0].ch_info_

    def _check_measures_fit(self):
        is_fit = True
        for meas in self.values():
            if not hasattr(meas, 'ch_info_'):
                is_fit = False
                break
        return is_fit

    def reduce_to_topo(self, measure_params, picks=None):
        if picks:  # XXX think if info is needed down-stream
            info = mne.io.pick.pick_info(self.info_, picks, copy=True)
        else:
            info = self.info_
        measures_to_topo = [meas for meas in self.values() if
                            isin_info(info_source=info,
                                      info_target=meas.info_)]
        n_measures, n_channels = len(measures_to_topo), info['nchan']
        out = np.empty((n_measures, n_channels), dtype=np.float64)
        for ii, meas in enumerate(measures_to_topo):
            out[ii] = meas.reduce_to_topo(info=info)
        return out

    def reduce_to_scalar(self, measure_params, picks=None):
        if picks:  # XXX think if info is needed down-stream
            info = mne.io.pick.pick_info(self.info_, picks, copy=True)
        else:
            info = self.info_
        n_measures = len(self)
        out = np.empty(n_measures, dtype=np.float64)
        for ii, meas in enumerate(self.values()):
            out[ii] = meas.reduce_to_scalar(info)

        return out

    def save(self, fname):
        write_hdf5(fname, self.keys(), title='nice/features/order')
        for meas in self.values():
            meas.save(fname)

    def add_measure(self, measure):
        self[measure._get_title()] = measure


def isin_info(info_source, info_target):
    set_diff_ch = len(set(info_source['ch_names']) -
                      set(info_target['ch_names']))
    is_compat = True
    if set_diff_ch > 0:
        is_compat = False
    return is_compat


def read_features(fname):
    measures_classes = dict(inspect.getmembers(sys.modules['nice.measures']))
    contents = h5_listdir(fname)
    measures = list()
    epochs = None
    if 'nice/features/order' in contents:
        measure_order = read_hdf5(fname, title='nice/features/order')
    else:
        measure_order = [k for k in contents if 'nice/measure/' in k]

    if any('nice/data/epochs' in k for k in contents):
        epochs = read_hdf5(fname, title='nice/data/epochs')
        epochs = mne.EpochsArray(
            data=epochs.pop('_data'), info=Info(epochs.pop('info')),
            tmin=epochs.pop('tmin'), event_id=epochs.pop('event_id'),
            events=epochs.pop('events'), reject=epochs.pop('reject'),
            flat=epochs.pop('flat'))
    for content in measure_order:
        _, _, my_class_name, comment = content.split('/')
        my_class = measures_classes[my_class_name]
        if issubclass(my_class, BaseEventRelated):
            if not epochs:
                raise RuntimeError(
                    'Something weird has happened. You want to read a '
                    'measure that depends on epochs but '
                    'I could not find any epochs in the file you gave me.')
            measures.append(my_class._read(fname, epochs, comment=comment))
        elif issubclass(my_class, BaseMeasure):
            measures.append(my_class._read(fname, comment=comment))
        else:
            raise ValueError('Come on--this is not a Nice class!')
    measures = Features(measures)
    return measures
