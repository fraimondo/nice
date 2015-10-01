from .base import BaseMeasure, _read_measure

import numpy as np
from mne.time_frequency import compute_epochs_psd


class PowerSpectralDensity(BaseMeasure):
    """docstring for PSD"""
    
    def __init__(self, fmin=0, fmax=np.inf, n_fft=256, n_overlap=0,
                 normalize=False, dB=True, n_jobs=1, comment='default'):
        self.fmin = fmin
        self.fmax = fmax
        self.normalize = normalize
        self.dB = dB
        self.n_jobs = n_jobs
        self.n_overlap = n_overlap
        self.n_fft = n_fft
        self.comment = comment

    def fit(self, epochs):
        # XXX XXX XXX (porny triple triple XXX)
        # check n_fft VS segment size in final MNE implementation ping @agramfort + yousra
        # XXX XXX XXX (porny triple triple XXX)
        psds, freqs = compute_epochs_psd(
            epochs=epochs, fmin=self.fmin, fmax=self.fmax,
            n_jobs=self.n_jobs, n_overlap=self.n_overlap,
            n_fft=self.n_fft)
        if self.normalize:
            psds /= psds.sum(axis=-1)[..., None]
            assert np.allclose(psds.sum(axis=-1), 1.)
        if self.dB is True and self.normalize is False:
            psds = 10 * np.log10(psds)
            unit = 'dB'
        else:
            unit = 'power'

        self.data_ = psds
        self.freqs_ = freqs
        self.unit_ = unit

    def reduction(self):
        # XXX spectral summaries
        pass


def read_psd(fname, comment='default'):
    return PowerSpectralDensity._read(fname, comment=comment)
