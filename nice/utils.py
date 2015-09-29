import os.path as op

import numpy as np
from scipy.io import loadmat

from mne import create_info
from mne.channels import read_montage
from mne.io import RawArray


def create_mock_data_egi(n_channels, n_samples, stim=True):
    """Load and configure testing data
    Parameters
    ----------
    n_channels : int
        The number of EEG channels.
    n_samples : int
        The number of time samples.
    stim : bool
        Whether to add a stim channel or not. Defaults to True.
    Returns
    -------
    raw : instance of mne.RawArry
        The testing data.
    """
    mat_contents = loadmat(
        op.join(op.realpath(op.dirname(__file__)),
                'io', 'tests', 'data', 'test-eeg.mat'))

    data = mat_contents['data'][:n_channels, :n_samples] * 1e-7
    sfreq = 250.
    if stim is True:
        ch_names = ['E%i' % i for i in range(1, n_channels + 1, 1)]
        ch_names += ['STI 014']
        ch_types = ['eeg'] * n_channels
        ch_types += ['stim']
        data = np.r_[data, data[-1:]]
        data[-1].fill(0)
    else:
        ch_names = ['E%i' % i for i in range(1, n_channels + 1, 1)]
        ch_types = ['eeg'] * n_channels

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=data, info=info)
    montage = read_montage('GSN-HydroCel-257')
    raw.set_montage(montage)
    info['description'] = 'geodesic256'
    return raw
