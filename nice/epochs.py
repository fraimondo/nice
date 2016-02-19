from mne.epochs import Epochs
from mne.time_frequency import compute_epochs_psd


class EpochsEnhancer(Epochs):
    def __init__(self, epochs, psds_params=None):
        for v in vars(epochs):
            setattr(self, v, getattr(epochs, v))
        # self.epochs_ = epochs
        self.psds_ = None
        if psds_params is None:
            psds_params = dict(tmin=None, tmax=None, n_fft=256, n_overlap=0,
                               n_jobs=1, fmin=0, fmax=epochs.info['sfreq'] / 2)
        self.psds_params_ = psds_params

    def get_psds(self):
        if self.psds_ is None:
            tmin = self.psds_params_.get('tmin', None)
            tmax = self.psds_params_.get('tmax', None)
            this_epochs = self.crop(tmin=tmin, tmax=tmax, copy=True)

            fmin = self.psds_params_.get('fmin', None)
            fmax = self.psds_params_.get('fmax', None)
            n_overlap = self.psds_params_.get('n_overlap', None)
            n_fft = self.psds_params_.get('n_fft', None)
            n_jobs = self.psds_params_.get('n_jobs', None)
            # XXX XXX XXX (porny triple triple XXX)
            # check n_fft VS segment size in final MNE implementation ping
            # @agramfort + yousra
            # XXX XXX XXX (porny triple triple XXX)
            self.psds_, self.freqs_ = compute_epochs_psd(
                epochs=this_epochs, fmin=fmin, fmax=fmax,
                n_jobs=n_jobs, n_overlap=n_overlap, n_fft=n_fft)
        return self.psds_, self.freqs_

    def _check_freq_range(self, fmin, fmax):
        this_max = self.psds_params_['fmax']
        if this_max is None:
            this_max = self.info['sfreq'] / 2

        this_min = self.psds_params_['fmin']
        if this_min is None:
            this_min = 0.

        in_range = fmin >= this_min and fmax <= this_max
        if not in_range:
            raise ValueError('Spectral parameters dont match')
