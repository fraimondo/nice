# License: BSD (3-clause)

import os.path as op

from nose.tools import assert_equal

from numpy.testing import assert_array_equal
import numpy as np
import warnings
import matplotlib

import mne
from mne.utils import _TempDir, clean_warning_registry

# our imports
from nice.measures import PowerSpectralDensity, read_psd
from nice.measures import ContingentNegativeVariation, read_cnv

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


def _base_io_test(inst, epochs, read_fun):
    tmp = _TempDir()
    inst.fit(epochs)
    inst.save(tmp + '/test.hdf5')
    inst2 = read_fun(tmp + '/test.hdf5')
    for k, v in vars(inst).items():
        v2 = getattr(inst2, k)
        if isinstance(v, np.ndarray):
            assert_array_equal(v, v2)
        else:
            assert_equal(v, v2)


def test_spectral():
    """Test computation of spectral measures"""
    epochs = _get_data()[:2]
    psd = PowerSpectralDensity(1, 4)
    _base_io_test(psd, epochs, read_psd)


def test_time_locked():
    """Test computation of spectral measures"""
    epochs = _get_data()[:2]
    cnv = ContingentNegativeVariation()
    _base_io_test(cnv, epochs, read_cnv)


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
