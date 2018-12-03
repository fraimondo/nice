# NICE
# Copyright (C) 2017 - Authors of NICE
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# You can be released from the requirements of the license by purchasing a
# commercial license. Buying such a license is mandatory as soon as you
# develop commercial activities as mentioned in the GNU Affero General Public
# License version 3 without disclosing the source code of your own
# applications.
#
from collections import OrderedDict
from .utils import h5_listdir
from mne.externals.h5io import read_hdf5, write_hdf5
from .markers.base import BaseMarker, BaseTimeLocked
from .markers.spectral import BasePowerSpectralDensity, read_psd_estimator
import sys
import inspect
import mne
from mne.io.meas_info import Info
from mne.utils import logger
# from mne.parallel import parallel_func
import numpy as np


class Markers(OrderedDict):

    def __init__(self, markers):
        OrderedDict.__init__(self)
        for meas in markers:
            self._add_marker(meas)
        if self._check_markers_fit():
            self.ch_info_ = list(self.values())[0].ch_info_

    def fit(self, epochs):
        for meas in self.values():
            if isinstance(meas, BasePowerSpectralDensity):
                meas._check_freq_time_range(epochs)

        for meas in self.values():
            logger.info('Fitting {}'.format(meas._get_title()))
            meas.fit(epochs)
        self.ch_info_ = list(self.values())[0].ch_info_

    def _check_markers_fit(self):
        is_fit = True
        for meas in self.values():
            if not hasattr(meas, 'ch_info_'):
                is_fit = False
                break
        return is_fit

    def topo_names(self):
        info = self.ch_info_
        marker_names = [
            meas._get_title() for meas in self.values() if isin_info(
                info_source=info, info_target=meas.ch_info_) and
            'channels' in meas._axis_map]
        return marker_names

    def scalar_names(self):
        return list(self.keys())

    def reduce_to_topo(self, marker_params):
        logger.info('Reducing to topographies')
        self._check_marker_params_keys(marker_params)
        ch_picks = mne.pick_types(self.ch_info_, eeg=True, meg=True)
        if ch_picks is not None:  # XXX think if info is needed down-stream
            info = mne.io.pick.pick_info(self.ch_info_, ch_picks, copy=True)
        else:
            info = self.ch_info_
        markers_to_topo = [
            meas for meas in self.values() if isin_info(
                info_source=info, info_target=meas.ch_info_) and
            'channels' in meas._axis_map]
        n_markers = len(markers_to_topo)
        if ch_picks is None:
            n_channels = info['nchan']
        else:
            n_channels = len(ch_picks)
        out = np.empty((n_markers, n_channels), dtype=np.float64)
        for ii, meas in enumerate(markers_to_topo):
            logger.info('Reducing {}'.format(meas._get_title()))
            this_params = _get_reduction_params(marker_params, meas)
            out[ii] = meas.reduce_to_topo(**this_params)
        return out

    def reduce_to_scalar(self, marker_params):
        logger.info('Reducing to scalars')
        self._check_marker_params_keys(marker_params)
        n_markers = len(self)
        out = np.empty(n_markers, dtype=np.float64)
        for ii, meas in enumerate(self.values()):
            logger.info('Reducing {}'.format(meas._get_title()))
            this_params = _get_reduction_params(marker_params, meas)
            out[ii] = meas.reduce_to_scalar(**this_params)

        return out

    def compress(self, reduction_func):
        for meas in self.values():
            if not isinstance(meas, BaseTimeLocked):
                meas.compress(reduction_func)

    def save(self, fname, overwrite=False):
        if not fname.endswith('-markers.hdf5'):
            logger.warning('Feature collections file name should end '
                           'with "-markers.hdf5". Some NICE markers '
                           'might not work.')
        write_hdf5(fname, list(self.keys()), title='nice/markers/order',
                   overwrite=overwrite, slash='replace')
        for meas in self.values():
            meas.save(fname, overwrite='update')

    def _add_marker(self, marker):
        self[marker._get_title()] = marker

    def add_marker(self, marker):
        if self._check_markers_fit():
            raise ValueError('Adding a marker to an already fit collection '
                             'is not allowed')
        if not isinstance(marker, list):
            marker = [marker]
        for meas in marker:
            self._add_marker(meas)

    def _check_marker_params_keys(self, marker_params):
        for key in marker_params.keys():
            error = False
            if '/' in key:
                klass, comment = key.split('/')
                if 'nice/marker/{}/{}'.format(klass, comment) not in self:
                    error = True
            else:
                prefix = 'nice/marker/{}'.format(key)
                if not any(k.startswith(prefix) for k in self.keys()):
                    error = True
            if error:
                raise ValueError('Your marker_params is inconsistent with '
                                 'the elements in this feature collection: '
                                 '{} is not a valid feature or class'
                                 .format(key))


# def _reduce_to(inst, target, params):
#     out = None
#     if target == 'topography':
#         out = inst.reduce_to_topo(**params)
#     elif target == 'scalar':
#         out = inst.reduce_to_scalar(**params)
#     return out


def _get_reduction_params(marker_params, meas):
    # XXX Check for typos and issue warnings
    full = '{}/{}'.format(meas.__class__.__name__, meas.comment)
    out = {}
    if full in marker_params:
        out = marker_params[full]
    else:
        part = full.split('/')[0]
        if part in marker_params:
            out = marker_params[part]
    if len(out) == 0:
        raise ValueError('No reduction for {}'.format(full))
    return out


def isin_info(info_source, info_target):
    set_diff_ch = len(set(info_source['ch_names']) -
                      set(info_target['ch_names']))
    is_compat = True
    if set_diff_ch > 0:
        is_compat = False
    return is_compat


def read_markers(fname):
    markers_classes = dict(inspect.getmembers(sys.modules['nice.markers']))
    contents = h5_listdir(fname)
    markers = list()
    epochs = None
    if 'nice/markers/order' in contents:
        marker_order = read_hdf5(fname, title='nice/markers/order',
                                 slash='replace')
    else:
        marker_order = [k for k in contents if 'nice/marker/' in k]

    if any('nice/data/epochs' in k for k in contents):
        epochs = read_hdf5(fname, title='nice/data/epochs', slash='replace')
        # MNE fix
        if 'filename' in epochs['info']:
            del(epochs['info']['filename'])
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
    for content in marker_order:
        _, _, my_class_name, comment = content.split('/')
        my_class = markers_classes[my_class_name]
        if issubclass(my_class, BaseTimeLocked):
            if not epochs:
                raise RuntimeError(
                    'Something weird has happened. You want to read a '
                    'marker that depends on epochs but '
                    'I could not find any epochs in the file you gave me.')
            markers.append(my_class._read(fname, epochs, comment=comment))
        elif issubclass(my_class, BasePowerSpectralDensity):
            markers.append(
                my_class._read(
                    fname, estimators=all_estimators, comment=comment))
        elif issubclass(my_class, BaseMarker):
            markers.append(my_class._read(fname, comment=comment))
        else:
            raise ValueError('Come on--this is not a Nice class!')
    markers = Markers(markers)
    return markers
