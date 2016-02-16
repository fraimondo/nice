from collections import OrderedDict
from .base import BaseMeasure

import numpy as np

from mne.utils import _time_mask as float_mask


class PowerSpectralDensity(BaseMeasure):
    """docstring for PSD"""

    def __init__(self, tmin=None, tmax=None, fmin=0, fmax=np.inf,
                 normalize=False, dB=True, comment='default'):
        BaseMeasure.__init__(self, tmin=None, tmax=None, comment=comment)
        self.fmin = fmin
        self.fmax = fmax
        self.normalize = normalize
        self.dB = dB

    @property
    def _axis_map(self):
        return OrderedDict([
            ('epochs', 0),
            ('channels', 1),
            ('frequency', 2)
        ])

    def _fit(self, epochs):
        epochs._check_freq_range(self.fmin, self.fmax)
        psds, freqs = epochs.get_psds()
        mask = float_mask(freqs, 1., 4.)
        this_psds = psds[:, :, mask]
        this_freqs = freqs[mask]
        if self.normalize:
            this_psds /= this_psds.sum(axis=-1)[..., None]
            assert np.allclose(this_psds.sum(axis=-1), 1.)
        if self.dB is True and self.normalize is False:
            this_psds = 10 * np.log10(this_psds)
            unit = 'dB'
        elif self.normalize:
            unit = 'perc'
        else:
            unit = 'power'

        self.data_ = psds
        self.freqs_ = this_freqs
        self.unit_ = unit


def read_psd(fname, comment='default'):
    return PowerSpectralDensity._read(fname, comment=comment)
