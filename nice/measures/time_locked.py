from .base import BaseMeasure, BaseEventRelated

from ..recipes.time_locked import epochs_compute_cnv

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
        return {
            'epochs': 0,
            'channels': 1
        }


def read_cnv(fname, comment='default'):
    return ContingentNegativeVariation._read(fname, comment=comment)


class EventRelatedTopography(BaseEventRelated):
    """docstring for ERP"""

    def __init__(self, tmin, tmax, subset=None, comment='default'):
        BaseEventRelated.__init__(self, tmin, tmax, comment)
        self.subset = subset

    @property
    def _axis_map(self):
        return {
            'epochs': 0,
            'channels': 1,
            'times': 2
        }

    def _prepare_data(self, picks):
        time_mask = _time_mask(self.epochs_.times, self.tmin, self.tmax)
        subset = self.subset
        picks = (pick_types(self.epochs_.info, eeg=True, meg=True)
                 if not picks else picks)
        return ((self.epochs_[subset] if subset else self.epochs_)
                .get_data()[:, picks][..., time_mask])


def read_ert(fname, epochs, comment='default'):
    return EventRelatedTopography._read(fname, epochs=epochs, comment=comment)


class EventRelatedContrast(BaseEventRelated):
    """docstring for ERP"""

    def __init__(self, tmin, tmax, condition_a, condition_b,
                 comment='default'):
        BaseEventRelated.__init__(self, tmin, tmax, comment)
        self.condition_a = condition_a
        self.condition_b = condition_b

    @property
    def _axis_map(self):
        return {
            'epochs': 0,
            'channels': 1,
            'times': 2
        }

    def _reduce_to(self, reduction_func, target, picks):
        cont_list = list()
        for cond in [self.condition_a, self.condition_b]:
            ert = EventRelatedTopography(self.tmin, self.tmax, subset=cond)
            ert.fit(self.epochs_)
            cont_list.append(ert._reduce_to(reduction_func, target, picks))
        return cont_list[0] - cont_list[1]


def read_erc(fname, epochs, comment='default'):
    return EventRelatedContrast._read(fname, epochs=epochs, comment=comment)
