from collections import OrderedDict
from .utils import h5_listdir
from .externals.h5io import read_hdf5, write_hdf5
from .measures.base import BaseMeasure, BaseEventRelated
import sys
import inspect
import mne
from mne.io.meas_info import Info


class Features(OrderedDict):

    def __init__(self, measures):
        OrderedDict.__init__(self)
        for meas in measures:
            self.add_measure(meas)

    def fit(self, epochs):
        for meas in self.values():
            meas.fit(epochs)

    def transform(self, measure_params, epochs=None):
        pass

    def save(self, fname):
        write_hdf5(fname, self.keys(), title='nice/features/order')
        for meas in self.values():
            meas.save(fname)

    def add_measure(self, measure):
        self[measure._get_title()] = measure


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
        if issubclass(my_class, BaseMeasure):
            measures.append(my_class._read(fname, comment=comment))
        if issubclass(my_class, BaseEventRelated):
            if not epochs:
                raise RuntimeError(
                    'Something weird has happened. You want to read a '
                    'measure that depends on epochs but '
                    'I could not find any epochs in the file you gave me.')
            measures.append(my_class._read(fname, epochs, comment=comment))
    measures = Features(measures)
    return measures
