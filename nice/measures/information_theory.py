from .base import BaseMeasure, _read_measure
from ..algorithms.information_theory import (epochs_compute_komplexity,
                                             epochs_compute_pe)


class Komplexity(BaseMeasure):
    """docstring for ContingentNegativeVariation"""

    def __init__(self, backend="python", nbins=32, method_params=None,
                 comment='default'):
        if method_params is None:
            method_params = {}

        self.nbins = nbins
        self.backend = backend
        self.method_params = method_params
        self.comment = comment

    def fit(self, epochs):
        komp = epochs_compute_komplexity(epochs, nbins=self.nbins,
                                         backend=self.backend,
                                         method_params=self.method_params)
        self.data_ = komp


def read_komplexity(fname, comment='default'):
    return _read_measure(Komplexity, fname, comment=comment)


class PermutationEntropy(BaseMeasure):
    """docstring for PermutationEntropy"""

    def __init__(self, kernel=3, tau=8, backend="python", comment='default'):
        self.kernel = kernel
        self.tau = tau
        self.backend = backend
        self.comment = comment

    def fit(self, epochs):
        pe, _ = epochs_compute_pe(epochs, kernel=self.kernel, tau=self.tau,
                                  backend=self.backend)
        self.data_ = pe


def read_pe(fname, comment='default'):
    return _read_measure(PermutationEntropy, fname, comment=comment)
