from .base import BaseMeasure, _read_measure
from ..externals.h5io import write_hdf5, read_hdf5
from ..algorithms.time_locked import epochs_compute_cnv

import numpy as np


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
