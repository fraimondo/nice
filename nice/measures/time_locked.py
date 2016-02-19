from collections import Counter, OrderedDict

import numpy as np

from .base import BaseMeasure, BaseTimeLocked, _read_measure

from ..recipes.time_locked import epochs_compute_cnv
from ..utils import mne_epochs_key_to_index
from ..algorithms.decoding import decode_window
from mne.utils import _time_mask
from mne.io.pick import pick_types


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

    def __init__(self, tmin, tmax, subset=None, comment='default'):
        BaseTimeLocked.__init__(self, tmin, tmax, comment)
        self.subset = subset

    @property
    def _axis_map(self):
        return OrderedDict([
            ('epochs', 0),
            ('channels', 1),
            ('times', 2)
        ])

    def _prepare_data(self, picks):
        time_mask = _time_mask(self.epochs_.times, self.tmin, self.tmax)
        subset = self.subset
        picks = (pick_types(self.epochs_.info, eeg=True, meg=True)
                 if not picks else picks)
        return ((self.epochs_[subset] if subset else self.epochs_)
                .get_data()[:, picks][..., time_mask])


def read_ert(fname, epochs, comment='default'):
    return TimeLockedTopography._read(fname, epochs=epochs, comment=comment)


class TimeLockedContrast(BaseTimeLocked):
    """docstring for ERP"""

    def __init__(self, tmin, tmax, condition_a, condition_b,
                 comment='default'):
        BaseTimeLocked.__init__(self, tmin, tmax, comment)
        self.condition_a = condition_a
        self.condition_b = condition_b

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
            ert = TimeLockedTopography(self.tmin, self.tmax, subset=cond)
            ert.fit(self.epochs_)
            cont_list.append(ert._reduce_to(reduction_func, target, picks))
        return cont_list[0] - cont_list[1]


def read_erc(fname, epochs, comment='default'):
    return TimeLockedContrast._read(fname, epochs=epochs, comment=comment)


class WindowDecoding(BaseMeasure):
    def __init__(self, tmin, tmax, condition_a, condition_b, decoding_params,
                 comment='default'):
        BaseMeasure.__init__(self, tmin, tmax, comment)
        self.condition_a = condition_a
        self.condition_b = condition_b
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
            ('scores', 0),
        ])

    def _prepare_window_decoding(self, epochs):
        count = Counter(epochs.events[:, 2])
        id_event = {v: k for k, v in epochs.event_id.items()}
        class_weights = {id_event[k]: 1. / v for k, v in count.items()}
        sample_weight = np.zeros(len(epochs.events), dtype=np.float)
        for k, v in epochs.event_id.items():
            this_index = epochs.events[:, 2] == v
            sample_weight[this_index] = class_weights[k]

        condition_a_mask = mne_epochs_key_to_index(epochs, self.condition_a)
        condition_b_mask = mne_epochs_key_to_index(epochs, self.condition_b)

        sample_weight_a = sample_weight[condition_a_mask]
        sample_weight_b = sample_weight[condition_b_mask]

        y = np.r_[np.zeros(condition_b_mask.sum()),
                  np.ones(condition_a_mask.sum())]

        X = np.concatenate([
            epochs.get_data()[condition_b_mask],
            epochs.get_data()[condition_a_mask]
        ]).reshape(len(y), -1)

        sample_weight = np.r_[sample_weight_b, sample_weight_a]
        return X, y, sample_weight


def read_wd(fname, comment='default'):
    return WindowDecoding._read(fname, comment=comment)


class TimeDecoding(BaseTimeLocked):
    def __init__(self, tmin, tmax, condition_a, condition_b, decoding_params,
                 comment='default'):
        BaseTimeLocked.__init__(self, tmin, tmax, comment)
        self.condition_a = condition_a
        self.condition_b = condition_b
