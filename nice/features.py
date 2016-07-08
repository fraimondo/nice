from collections import OrderedDict
from .utils import h5_listdir
from .externals.h5io import read_hdf5, write_hdf5
from .measures.base import BaseMeasure, BaseTimeLocked
from .measures.spectral import BasePowerSpectralDensity, read_psd_estimator
import sys
import inspect
import mne
from mne.io.meas_info import Info
from mne.utils import logger
# from mne.parallel import parallel_func
import numpy as np


class Features(OrderedDict):

    def __init__(self, measures):
        OrderedDict.__init__(self)
        for meas in measures:
            self._add_measure(meas)
        if self._check_measures_fit():
            self.ch_info_ = list(self.values())[0].ch_info_

    def fit(self, epochs):
        for meas in self.values():
            if isinstance(meas, BasePowerSpectralDensity):
                meas._check_freq_time_range(epochs)

        for meas in self.values():
            logger.info('Fitting {}'.format(meas._get_title()))
            meas.fit(epochs)
        self.ch_info_ = list(self.values())[0].ch_info_

    def _check_measures_fit(self):
        is_fit = True
        for meas in self.values():
            if not hasattr(meas, 'ch_info_'):
                is_fit = False
                break
        return is_fit

    def topo_names(self):
        info = self.ch_info_
        measure_names = [
            meas._get_title() for meas in self.values() if isin_info(
                info_source=info, info_target=meas.ch_info_) and
            'channels' in meas._axis_map]
        return measure_names

    def reduce_to_topo(self, measure_params):
        logger.info('Reducing to topographies')
        # if n_jobs == 'auto':
        #     try:
        #         import multiprocessing as mp
        #         n_jobs = mp.cpu_count()
        #         logger.info(
        #             'Autodetected number of jobs {}'.format(n_jobs))
        #     except:
        #         logger.info('Cannot autodetect number of jobs')
        #         n_jobs = 1

        self._check_measure_params_keys(measure_params)
        ch_picks = self._check_measure_params_picks(measure_params)
        ch_picks = mne.pick_types(self.ch_info_, eeg=True, meg=True)
        if ch_picks is not None:  # XXX think if info is needed down-stream
            info = mne.io.pick.pick_info(self.ch_info_, ch_picks, copy=True)
        else:
            info = self.ch_info_
        measures_to_topo = [
            meas for meas in self.values() if isin_info(
                info_source=info, info_target=meas.ch_info_) and
            'channels' in meas._axis_map]
        n_measures = len(measures_to_topo)
        if ch_picks is None:
            n_channels = info['nchan']
        else:
            n_channels = len(ch_picks)
        out = np.empty((n_measures, n_channels), dtype=np.float64)
        # parallel, _reduce_to_topo, _ = parallel_func(_reduce_to, n_jobs)
        # out = np.asarray(parallel(_reduce_to_topo(
        #     meas, target='topography',
        #     params=_get_reduction_params(measure_params, meas)
        # ) for meas in measures_to_topo))
        for ii, meas in enumerate(measures_to_topo):
            logger.info('Reducing {}'.format(meas._get_title()))
            this_params = _get_reduction_params(measure_params, meas)
            out[ii] = meas.reduce_to_topo(**this_params)
        return out

    def reduce_to_scalar(self, measure_params):
        logger.info('Reducing to scalars')
        # if n_jobs == 'auto':
        #     try:
        #         import multiprocessing as mp
        #         n_jobs = mp.cpu_count()
        #         logger.info(
        #             'Autodetected number of jobs {}'.format(n_jobs))
        #     except:
        #         logger.info('Cannot autodetect number of jobs')
        #         n_jobs = 1

        self._check_measure_params_keys(measure_params)
        # if picks:  # XXX think if info is needed down-stream
        #     info = mne.io.pick.pick_info(self.ch_info_, picks, copy=True)
        # else:
            # info = self.ch_info_
        n_measures = len(self)
        out = np.empty(n_measures, dtype=np.float64)
        # parallel, _reduce_to_scalar, _ = parallel_func(_reduce_to, n_jobs)
        # out = np.asarray(parallel(_reduce_to_scalar(
        #     meas, target='scalar',
        #     params=_get_reduction_params(measure_params, meas)
        # ) for meas in self.values()))
        for ii, meas in enumerate(self.values()):
            logger.info('Reducing {}'.format(meas._get_title()))
            this_params = _get_reduction_params(measure_params, meas)
            out[ii] = meas.reduce_to_scalar(**this_params)

        return out

    def compress(self, reduction_func):
        for meas in self.values():
            if not isinstance(meas, BaseTimeLocked):
                meas.compress(reduction_func)

    def save(self, fname, overwrite=False):
        write_hdf5(fname, list(self.keys()), title='nice/features/order',
                   overwrite=overwrite)
        for meas in self.values():
            meas.save(fname, overwrite='update')

    def _add_measure(self, measure):
        self[measure._get_title()] = measure

    def add_measure(self, measure):
        if self._check_measures_fit():
            raise ValueError('Adding a measure to an already fit collection '
                             'is not allowed')
        if not isinstance(measure, list):
            measure = [measure]
        for meas in measure:
            self._add_measure(meas)

    def _check_measure_params_keys(self, measure_params):
        for key in measure_params.keys():
            error = False
            if '/' in key:
                klass, comment = key.split('/')
                if 'nice/measure/{}/{}'.format(klass, comment) not in self:
                    error = True
            else:
                prefix = 'nice/measure/{}'.format(key)
                if not any(k.startswith(prefix) for k in self.keys()):
                    error = True
            if error:
                raise ValueError('Your measure_params is inconsistent with '
                                 'the elements in this feature collection: '
                                 '{} is not a valid feature or class'
                                 .format(key))

    def _check_measure_params_picks(self, measure_params):
        # Check that if we pick channels, we do it the same way on every
        # measure
        all_picks = []
        for key, params in measure_params.items():
            if 'picks' in params:
                picks = params['picks']
                if 'channels' in picks:
                    all_picks.append(np.sort(picks['channels']))

        first = None
        if len(all_picks) > 0:
            first = all_picks[0]
        equal = True
        for this_picks in all_picks[1:]:
            equal = np.all(this_picks == first)
            if not equal:
                raise ValueError('Your picks are inconsistent among each other '
                                 'element in this feature collection. Channels '
                                 'picks should be the same')
        return first


# def _reduce_to(inst, target, params):
#     out = None
#     if target == 'topography':
#         out = inst.reduce_to_topo(**params)
#     elif target == 'scalar':
#         out = inst.reduce_to_scalar(**params)
#     return out


def _get_reduction_params(measure_params, meas):
    # XXX Check for typos and issue warnings
    full = '{}/{}'.format(meas.__class__.__name__, meas.comment)
    out = {}
    if full in measure_params:
        out = measure_params[full]
    else:
        part = full.split('/')[0]
        if part in measure_params:
            out = measure_params[part]
    return out


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
    # Read all PowerSpectralDensityEstimator estimators
    estimators = [k for k in contents if
                  'nice/container/PowerSpectralDensityEstimator' in k]
    all_estimators = {}
    for estimator_name in estimators:
        estimator_comment = estimator_name.split('/')[-1]
        this_estimator = read_psd_estimator(fname, comment=estimator_comment)
        all_estimators[estimator_comment] = this_estimator
    for content in measure_order:
        _, _, my_class_name, comment = content.split('/')
        my_class = measures_classes[my_class_name]
        if issubclass(my_class, BaseTimeLocked):
            if not epochs:
                raise RuntimeError(
                    'Something weird has happened. You want to read a '
                    'measure that depends on epochs but '
                    'I could not find any epochs in the file you gave me.')
            measures.append(my_class._read(fname, epochs, comment=comment))
        elif issubclass(my_class, BasePowerSpectralDensity):
            measures.append(
                my_class._read(
                    fname, estimators=all_estimators, comment=comment))
        elif issubclass(my_class, BaseMeasure):
            measures.append(my_class._read(fname, comment=comment))
        else:
            raise ValueError('Come on--this is not a Nice class!')
    measures = Features(measures)
    return measures
