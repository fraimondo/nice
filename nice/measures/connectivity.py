from .base import BaseMeasure
from ..algorithms.connectivity import epochs_compute_wsmi


class SymbolicMutualInformation(BaseMeasure):
    """docstring for SymbolicMutualInformation"""

    def __init__(self, kernel=3, tau=8, backend="python", method_params=None,
                 method='weighted', comment='default'):
        if method_params is None:
            method_params = {}
        self.kernel = kernel
        self.tau = tau
        self.backend = backend
        self.method_params = method_params
        self.comment = comment
        self.method = method

    def _fit(self, epochs):
        wsmi, smi, _, _ = epochs_compute_wsmi(epochs, kernel=self.kernel,
                                              tau=self.tau,
                                              backend=self.backend,
                                              method_params=self.method_params)
        self.data_ = wsmi if self.method == 'weighted' else smi

    @property
    def _axis_map(self):
        return {
            'epochs': 0,
            'channels': 1,
            'channels_y': 2,
        }


def read_wsmi(fname, comment='default'):
    return SymbolicMutualInformation._read(fname, comment=comment)
