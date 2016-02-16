from collections import OrderedDict
import numpy as np

from .base import BaseMeasure
from ..algorithms.connectivity import epochs_compute_wsmi


class SymbolicMutualInformation(BaseMeasure):
    """docstring for SymbolicMutualInformation"""

    def __init__(self, tmin=None, tmax=None, kernel=3, tau=8, backend="python",
                 method_params=None, method='weighted', comment='default'):
        BaseMeasure.__init__(self, tmin, tmax, comment)
        if method_params is None:
            method_params = {}
        self.kernel = kernel
        self.tau = tau
        self.backend = backend
        self.method_params = method_params
        self.method = method

    def _fit(self, epochs):
        wsmi, smi, _, _ = epochs_compute_wsmi(
            epochs, kernel=self.kernel, tau=self.tau, tmin=self.tmin,
            tmax=self.tmax, backend=self.backend,
            method_params=self.method_params)
        data = wsmi if self.method == 'weighted' else smi
        data += np.transpose(data, [1, 0, 2])
        self.data_ = data

    @property
    def _axis_map(self):
        return OrderedDict([
            ('channels', 0),
            ('channels_y', 1),
            ('epochs', 2)
        ])


def read_smi(fname, comment='default'):
    return SymbolicMutualInformation._read(fname, comment=comment)
