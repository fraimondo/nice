from collections import Counter, OrderedDict

import numpy as np

from .base import (BaseMeasure, BaseTimeLocked, _read_container)

from ..recipes.time_locked import epochs_compute_cnv
from ..utils import mne_epochs_key_to_index, epochs_has_event
from ..algorithms.decoding import decode_window
from mne.utils import _time_mask
from mne.io.pick import pick_types
import mne.decoding as mne_decoding


class ContingentNegativeVariation(BaseMeasure):
    """docstring for ContingentNegativeVariation"""

    def __init__(self, tmin=None, tmax=None, comment='default'):
        BaseMeasure.__init__(self, tmin, tmax, comment)

    def _fit(self, epochs):
        cnv = epochs_compute_cnv(epochs, self.tmin, self.tmax)
        self.data_ = cnv

    @property
    def _axis_map(self):
        return OrderedDict([
            ('epochs', 0),
            ('channels', 1)
        ])


def read_cnv(fname, comment='default'):
    return ContingentNegativeVariation._read(fname, comment=comment)


class TimeLockedTopography(BaseTimeLocked):
    """docstring for ERP"""

    def __init__(self, tmin, tmax, subset=None, missing_nan=False,
                 comment='default'):
        BaseTimeLocked.__init__(self, tmin, tmax, comment)
        self.subset = subset
        self.missing_nan = missing_nan

    @property
    def _axis_map(self):
        return OrderedDict([
            ('epochs', 0),
            ('channels', 1),
            ('times', 2)
        ])

    def _prepare_data(self, picks, target):
        this_picks = {k: None for k in ['times', 'channels', 'epochs']}
        if picks is not None:
            if any([x not in this_picks.keys() for x in picks.keys()]):
                raise ValueError('Picking is not compatible for {}'.format(
                    self._get_title()))
        if picks is None:
            picks = {}
        this_picks.update(picks)
        to_preserve = self._get_preserve_axis(target)
        if len(to_preserve) > 0:
            for axis in to_preserve:
                this_picks[axis] = None

        # Pick Times based on original times
        time_picks = this_picks['times']
        time_mask = _time_mask(self.epochs_.times, self.tmin, self.tmax)
        if time_picks is not None:
            picks_mask = np.zeros(len(time_mask), dtype=np.bool)
            picks_mask[time_picks] = True
            time_mask = np.logical_and(time_mask, picks_mask)

        # Pick epochs based on original indices
        epochs_picks = this_picks['epochs']
        this_epochs = self.epochs_
        if epochs_picks is not None:
            this_epochs = this_epochs[epochs_picks]

        # Pick channels based on original indices
        ch_picks = this_picks['channels']
        if ch_picks is None:
            ch_picks = pick_types(this_epochs.info, eeg=True, meg=True)

        if (self.subset and self.missing_nan and not
                epochs_has_event(this_epochs, self.subset)):
            data = np.array([[[np.nan]]])
        else:
            if self.subset:
                this_epochs = this_epochs[self.subset]
            data = this_epochs.get_data()[:, ch_picks][..., time_mask]

        return data


def read_ert(fname, epochs, comment='default'):
    return TimeLockedTopography._read(fname, epochs=epochs, comment=comment)


class TimeLockedContrast(BaseTimeLocked):
    """docstring for ERP"""

    def __init__(self, tmin, tmax, condition_a, condition_b, missing_nan=False,
                 comment='default'):
        BaseTimeLocked.__init__(self, tmin, tmax, comment)
        self.condition_a = condition_a
        self.condition_b = condition_b
        self.missing_nan = missing_nan

    @property
    def _axis_map(self):
        return OrderedDict([
            ('epochs', 0),
            ('channels', 1),
            ('times', 2)
        ])

    def _reduce_to(self, reduction_func, target, picks):
        cont_list = list()
        for cond in [self.condition_a, self.condition_b]:
            ert = TimeLockedTopography(self.tmin, self.tmax, subset=cond,
                                       missing_nan=self.missing_nan)
            ert.fit(self.epochs_)
            cont_list.append(ert._reduce_to(reduction_func, target, picks))
        return cont_list[0] - cont_list[1]


def read_erc(fname, epochs, comment='default'):
    return TimeLockedContrast._read(fname, epochs=epochs, comment=comment)


class WindowDecoding(BaseMeasure):
    def __init__(self, tmin, tmax, condition_a, condition_b,
                 decoding_params=None, comment='default'):
        BaseMeasure.__init__(self, tmin, tmax, comment)
        self.condition_a = condition_a
        self.condition_b = condition_b
        if decoding_params is None:
            decoding_params = dict(
                sample_weight='auto',
                n_jobs='auto',
                cv=None,
                clf=None,
                labels=None,
                random_state=None,
            )
        self.decoding_params = decoding_params

    def _fit(self, epochs):
        dp = self.decoding_params

        X, y, sample_weight = self._prepare_window_decoding(epochs)
        if dp['sample_weight'] not in ('auto', None):
            sample_weight = dp['sample_weight']

        probas, predictions, scores = decode_window(
            X, y, clf=dp['clf'], cv=dp['cv'],
            sample_weight=sample_weight, n_jobs=dp['n_jobs'],
            random_state=dp['random_state'], labels=dp['labels'])
        self.data_ = scores
        self.other_outputs_ = {'probas': probas, 'predictions': predictions}
        self.shape_ = self.data_.shape

    @property
    def _axis_map(self):
        return OrderedDict([
            ('folds', 0),
        ])

    def _prepare_window_decoding(self, epochs):
        y, condition_a_mask, condition_b_mask = _prepare_y(
            epochs, self.condition_a, self.condition_b)
        sample_weight = _prepare_sample_weights(
            epochs, condition_a_mask, condition_b_mask)
        X = np.concatenate([
            epochs.get_data()[condition_b_mask],
            epochs.get_data()[condition_a_mask]
        ]).reshape(len(y), -1)

        return X, y, sample_weight


def read_wd(fname, comment='default'):
    return WindowDecoding._read(fname, comment=comment)


class TimeDecoding(BaseMeasure):
    def __init__(self, tmin, tmax, condition_a, condition_b, decoding_params,
                 comment='default'):
        BaseMeasure.__init__(self, tmin, tmax, comment)
        self.condition_a = condition_a
        self.condition_b = condition_b
        self.decoding_params = decoding_params

    def _fit(self, epochs):
        dp = {k: v for k, v in self.decoding_params.items()}
        dp['times'] = dict(start=self.tmin, stop=self.tmax)
        td = mne_decoding.TimeDecoding(**dp)
        y, _, _ = _prepare_y(epochs, self.condition_a, self.condition_b)
        td.fit(epochs, y=y)
        td.score(epochs, y=y)
        self.data_ = np.array(td.scores_)
        self.shape_ = self.data_.shape

    @property
    def _axis_map(self):
        return OrderedDict([
            ('times', 0),
        ])


def read_td(fname, comment='default'):
    return TimeDecoding._read(fname, comment=comment)


class GeneralizationDecoding(BaseMeasure):
    def __init__(self, tmin, tmax, condition_a, condition_b, decoding_params,
                 comment='default'):
        BaseMeasure.__init__(self, tmin, tmax, comment)
        self.condition_a = condition_a
        self.condition_b = condition_b
        self.decoding_params = decoding_params

    def _fit(self, epochs):
        dp = {k: v for k, v in self.decoding_params.items()}
        dp['train_times'] = dict(start=self.tmin, stop=self.tmax)
        dp['test_times'] = dict(start=self.tmin, stop=self.tmax)
        td = mne_decoding.GeneralizationAcrossTime(**dp)
        y, _, _ = _prepare_y(epochs, self.condition_a, self.condition_b)
        td.fit(epochs, y=y)
        td.score(epochs, y=y)
        self.data_ = np.array(td.scores_)
        self.shape_ = self.data_.shape

    @property
    def _axis_map(self):
        return OrderedDict([
            ('train_times', 0),
            ('test_times', 1),
        ])


def read_gd(fname, comment='default'):
    return GeneralizationDecoding._read(fname, comment=comment)


def _prepare_sample_weights(epochs, condition_a_mask, condition_b_mask):
    count = Counter(epochs.events[:, 2])
    id_event = {v: k for k, v in epochs.event_id.items()}
    class_weights = {id_event[k]: 1. / v for k, v in count.items()}
    sample_weight = np.zeros(len(epochs.events), dtype=np.float)
    for k, v in epochs.event_id.items():
        this_index = epochs.events[:, 2] == v
        sample_weight[this_index] = class_weights[k]

    sample_weight_a = sample_weight[condition_a_mask]
    sample_weight_b = sample_weight[condition_b_mask]
    sample_weight = np.r_[sample_weight_b, sample_weight_a]
    return sample_weight


def _prepare_y(epochs, condition_a, condition_b):
    condition_a_mask = mne_epochs_key_to_index(epochs, condition_a)
    condition_b_mask = mne_epochs_key_to_index(epochs, condition_b)

    y = np.r_[np.zeros(condition_b_mask.shape[0]),
              np.ones(condition_a_mask.shape[0])]

    return y, condition_a_mask, condition_b_mask
