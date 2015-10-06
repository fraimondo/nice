from .base import BaseMeasure, BaseEventRelated

from ..recipes.time_locked import epochs_compute_cnv

from mne.utils import _time_mask


class ContingentNegativeVariation(BaseMeasure):
    """docstring for ContingentNegativeVariation"""

    def __init__(self, tmin=None, tmax=None, comment='default'):
        self.tmin = tmin
        self.tmax = tmax
        self.comment = comment

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
        self.tmin = tmin
        self.tmax = tmax
        self.comment = comment
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
        picks = Ellipsis if not picks else picks
        return ((self.epochs_[subset] if subset else self.epochs)
                .get_data()[..., picks, time_mask])


def read_ert(fname, epochs, comment='default'):
    return EventRelatedTopography._read(fname, epochs=epochs, comment=comment)


class EventRelatedContrast(BaseEventRelated):
    """docstring for ERP"""

    def __init__(self, tmin, tmax, condition_a, condition_b,
                 summary_function='np.mean', comment='default'):
        self.tmin = tmin
        self.tmax = tmax
        self.summary_function = summary_function
        self.condition_a = condition_a
        self.condition_b = condition_b
        self.comment = comment

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
