from .base import BaseMeasure, _read_measure
from ..algorithms.connectivity import epochs_compute_wsmi

class WSMI(BaseMeasure):
    """docstring for WSMI"""

    def __init__(self, kernel=3, tau=8, backend="python", method_params=None,
                 comment='default'):
        if method_params is None:
            method_params = {}
        self.kernel = kernel
        self.tau = tau
        self.backend = backend
        self.method_params = method_params
        self.comment = comment

    def fit(self, epochs):
        wsmi, smi, _, _ = epochs_compute_wsmi(epochs, kernel=self.kernel,
                                              tau=self.tau,
                                              backend=self.backend,
                                              method_params=self.method_params)
        self.wsmi_ = wsmi
        self.smi_ = smi


def read_wsmi(fname, comment='default'):
    return _read_measure(WSMI, fname, comment=comment)
