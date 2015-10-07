from .base import BaseMeasure
from ..algorithms.information_theory import (epochs_compute_komplexity,
                                             epochs_compute_pe)


class KolmogorovComplexity(BaseMeasure):
    """docstring for ContingentNegativeVariation"""

    def __init__(self, tmin=None, tmax=None, backend="python", nbins=32,
                 method_params=None, comment='default'):
        BaseMeasure.__init__(self, tmin, tmax, comment)
        if method_params is None:
            method_params = {}

        self.nbins = nbins
        self.backend = backend
        self.method_params = method_params

    def _fit(self, epochs):
        komp = epochs_compute_komplexity(
            epochs, nbins=self.nbins, tmin=self.tmin,
            tmax=self.tmax, backend=self.backend,
            method_params=self.method_params)
        self.data_ = komp

    @property
    def _axis_map(self):
        return {
            'channels': 0,
            'epochs': 1,
        }


def read_komplexity(fname, comment='default'):
    return KolmogorovComplexity._read(fname, comment=comment)


class PermutationEntropy(BaseMeasure):
    """docstring for PermutationEntropy"""

    def __init__(self, tmin=None, tmax=None, kernel=3, tau=8, backend="python",
                 comment='default'):
        BaseMeasure.__init__(self, tmin, tmax, comment)
        self.kernel = kernel
        self.tau = tau
        self.backend = backend

    def _fit(self, epochs):
        pe, _ = epochs_compute_pe(
            epochs, tmin=self.tmin, tmax=self.tmax,
            kernel=self.kernel, tau=self.tau, backend=self.backend)
        self.data_ = pe

    @property
    def _axis_map(self):
        return {
            'channels': 0,
            'epochs': 1,
        }


def read_pe(fname, comment='default'):
    return PermutationEntropy._read(fname, comment=comment)
