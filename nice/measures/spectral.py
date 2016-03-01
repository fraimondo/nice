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
        mask = float_mask(freqs, self.fmin, self.fmax)
        this_psds = psds[:, :, mask]
        this_freqs = freqs[mask]
        if self.normalize:
            this_psds = this_psds / psds.sum(axis=-1)[..., None]
            # assert np.allclose(this_psds.sum(axis=-1), 1.)
        if self.dB is True and self.normalize is False:
            this_psds = 10 * np.log10(this_psds)
            unit = 'dB'
        elif self.normalize:
            unit = 'perc'
        else:
            unit = 'power'

        self.data_ = this_psds
        self.freqs_ = this_freqs
        self.unit_ = unit


def read_psd(fname, comment='default'):
    return PowerSpectralDensity._read(fname, comment=comment)


class PowerSpectralDensitySummary(BaseMeasure):
    """docstring for PSD"""

    def __init__(self, percentile, tmin=None, tmax=None, fmin=0, fmax=np.inf,
                 comment='default'):
        BaseMeasure.__init__(self, tmin=None, tmax=None, comment=comment)
        self.fmin = fmin
        self.fmax = fmax
        self.percentile = percentile

    @property
    def _axis_map(self):
        return OrderedDict([
            ('epochs', 0),
            ('channels', 1)
        ])

    def _fit(self, epochs):
        epochs._check_freq_range(self.fmin, self.fmax)
        psds, freqs = epochs.get_psds()
        mask = float_mask(freqs, self.fmin, self.fmax)
        this_psds = psds[:, :, mask]
        this_freqs = freqs[mask]
        this_psds = this_psds / this_psds.sum(axis=-1)[..., None]

        cumulative_spectra = np.cumsum(this_psds, axis=-1)
        idx = np.argmin((cumulative_spectra - self.percentile) ** 2, axis=-1)

        if psds.ndim > 2:
            data = np.zeros_like(idx, dtype=np.float)
            for iepoch in range(cumulative_spectra.shape[0]):
                data[iepoch] = freqs[idx[iepoch]]
        else:
            data = this_freqs[idx]
        self.data_ = data
        self.freqs_ = this_freqs
        self.unit_ = 'Hz'


def read_psds(fname, comment='default'):
    return PowerSpectralDensitySummary._read(fname, comment=comment)
