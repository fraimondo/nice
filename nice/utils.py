import os.path as op

import numpy as np
from scipy.io import loadmat
import h5py

from mne import create_info
from mne.channels import read_montage
from mne.io import RawArray
from .externals.h5io import write_hdf5


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


def h5_listdir(fname, max_depth=4):

    datasets = list()

    def _get_dataset(name, member):
        if isinstance(member, h5py.Dataset):
            name_ = '/'.join(name.split('/', max_depth)[:-1])
            if name_ not in datasets:
                datasets.append(name_)

    with h5py.File(fname, 'r') as fid:
        fid.visititems(_get_dataset)

    return datasets


def write_hdf5_mne_epochs(fname, epochs):
    epochs_vars = {k: v for k, v in vars(epochs).items() if
                   not k.startswith('_') or k == '_data'}
    write_hdf5(fname, epochs_vars,
               title='nice/data/epochs')
