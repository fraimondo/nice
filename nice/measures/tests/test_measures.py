# License: BSD (3-clause)

import os.path as op
from copy import deepcopy

from nose.tools import (assert_true, assert_equal, assert_raises,
                        assert_not_equal)

from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_allclose)
import numpy as np
import copy as cp
import warnings
from scipy import fftpack
import matplotlib

import mne
from mne.utils import _TempDir, clean_warning_registry

from mne.externals.six import text_type
from mne.externals.six.moves import zip, cPickle as pickle

# our imports
from nice.measures import PowerSpectralDensity, read_psd

matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = 2
preload = True


def _get_data():
    raw = mne.io.Raw(raw_fname, add_eeg_ref=False, proj=False)
    events = mne.read_events(event_name)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True,
                           ecg=True, eog=True, include=['STI 014'],
                           exclude='bads')[::15]

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        preload=preload, decim=3)
    return epochs


clean_warning_registry()  # really clean warning stack


def test_spectral():
    """Test computation of spectral measures"""
    tpm = _TempDir()
    epochs = _get_data()[:2]
    psd = PowerSpectralDensity(1, 4)
    psd.fit(epochs)
    psd.save(tpm + '/test.hdf5')
    psd2 = read_psd(tpm + '/test.hdf5')
    for k, v in vars(psd).items():
        v2 = getattr(psd2, k)
        if isinstance(v, np.ndarray):
            assert_array_equal(v, v2)
        else:
            assert_equal(v, v2)


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
