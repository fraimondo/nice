from .base import (
    BaseMeasure, BaseEventRelated, _read_measure, _check_epochs_consistency)
from ..recipes.time_locked import epochs_compute_cnv


class ContingentNegativeVariation(BaseMeasure):
    """docstring for ContingentNegativeVariation"""

    def __init__(self, tmin=None, tmax=None, comment='default'):
        self.tmin = tmin
        self.tmax = tmax
        self.comment = comment

    def fit(self, epochs):
        cnv = epochs_compute_cnv(epochs, self.tmin, self.tmax)
        self.data_ = cnv


def read_cnv(fname, comment='default'):
    return _read_measure(ContingentNegativeVariation, fname, comment=comment)


class EventRelatedTopography(BaseEventRelated):
    """docstring for ERP"""

    def __init__(self, tmin, tmax, summary=np.mean):
        pass


class EventRelatedContrast(BaseEventRelated):
    """docstring for ERP"""

    def __init__(self, arg):
        pass


def read_ert(fname, epochs, comment='default'):
    out = _read_measure(EventRelatedTopography, fname, comment=comment)
    out.data_ = _check_epochs_consistency(out.epochs_info_, epochs)
    return out


def read_erc(fname, epochs, comment='default'):
    out = _read_measure(EventRelatedContrast, fname, comment=comment)
    out.data_ = _check_epochs_consistency(out, epochs)
    return out
