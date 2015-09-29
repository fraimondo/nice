from .base import BaseMeasure, _read_measure
from ..algorithms.information_theory import epochs_compute_komplexity

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
        komp = epochs_compute_komplexity(epochs, self.nbins, self.backend,
                                         self.method_params)
        self.data_ = komp


def read_komplexity(fname, comment='default'):
    return _read_measure(Komplexity, fname, comment=comment)
