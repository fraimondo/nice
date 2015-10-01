from .base import BaseMeasure, BaseEventRelated

from ..recipes.time_locked import epochs_compute_cnv


class ContingentNegativeVariation(BaseMeasure):
    """docstring for ContingentNegativeVariation"""

    def __init__(self, tmin=None, tmax=None, comment='default'):
        self.tmin = tmin
        self.tmax = tmax
        self.comment = comment

    def _fit(self, epochs):
        cnv = epochs_compute_cnv(epochs, self.tmin, self.tmax)
        self.data_ = cnv


def read_cnv(fname, comment='default'):
    return ContingentNegativeVariation._read(fname, comment=comment)


class EventRelatedTopography(BaseEventRelated):
    """docstring for ERP"""

    def __init__(self, tmin, tmax, summary_function='np.mean',
                 comment='default'):
        self.tmin = tmin
        self.tmax = tmax
        self.summary_function = summary_function
        self.comment = comment


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


def read_erc(fname, epochs, comment='default'):
    return EventRelatedContrast._read(fname, epochs=epochs, comment=comment)
