# License: BSD (3-clause)

import os.path as op

from nose.tools import assert_equal, assert_true

from numpy.testing import assert_array_equal
import numpy as np
import warnings
import matplotlib

import mne
import h5py
from mne.utils import _TempDir, clean_warning_registry

# our imports
from nice.measures import PowerSpectralDensity, read_psd
from nice.measures import ContingentNegativeVariation, read_cnv
from nice.measures import KolmogorovComplexity, read_komplexity
from nice.measures import PermutationEntropy, read_pe
from nice.measures import SymbolicMutualInformation, read_wsmi

from nice.measures import EventRelatedTopography, read_ert

from nice.measures import EventRelatedContrast, read_erc

matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')

event_id, tmin, tmax = 1, -0.2, 0.5
event_id_2 = {'a': 1, 'b': 2}
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


def _compare_instance(inst1, inst2):
    for k, v in vars(inst1).items():
        v2 = getattr(inst2, k)
        if k == 'ch_info_' and v2 is None:
            continue
        if isinstance(v, np.ndarray):
            assert_array_equal(v, v2)
        elif isinstance(v, mne.io.meas_info.Info):
            pass
        else:
            assert_equal(v, v2)


def _base_io_test(inst, epochs, read_fun):
    tmp = _TempDir()
    inst.fit(epochs)
    inst.save(tmp + '/test.hdf5')
    inst2 = read_fun(tmp + '/test.hdf5')
    _compare_instance(inst, inst2)


def _erfp_io_test(tmp, inst, epochs, read_fun, comment='default'):
    inst.fit(epochs)
    inst.save(tmp + '/test.hdf5')
    inst2 = read_fun(tmp + '/test.hdf5', epochs, comment=comment)
    assert_array_equal(inst.epochs_.get_data(), inst2.epochs_.get_data())
    _compare_instance(inst, inst2)


def test_spectral():
    """Test computation of spectral measures"""
    epochs = _get_data()[:2]
    psd = PowerSpectralDensity(1, 4)
    _base_io_test(psd, epochs, read_psd)


def test_time_locked():
    """Test computation of time lockedi measures"""

    epochs = _get_data()[:2]
    cnv = ContingentNegativeVariation()
    _base_io_test(cnv, epochs, read_cnv)

    tmp = _TempDir()
    with h5py.File(tmp + '/test.hdf5') as fid:
        assert_true('nice/data/epochs' not in fid)
    ert = EventRelatedTopography(0.1, 0.2)
    _erfp_io_test(tmp, ert, epochs, read_ert)
    with h5py.File(tmp + '/test.hdf5') as fid:
        assert_true(fid['nice/data/epochs'].keys() != [])

    tmp = _TempDir()
    with h5py.File(tmp + '/test.hdf5') as fid:
        assert_true('nice/data/epochs' not in fid)
    erc = EventRelatedContrast(0.1, 0.2, 'a', 'b')
    _erfp_io_test(tmp, erc, epochs, read_erc)
    with h5py.File(tmp + '/test.hdf5') as fid:
        assert_true('nice/data/epochs' in fid)
    erc = EventRelatedContrast(0.1, 0.3, 'a', 'b', comment='another_erp')
    _erfp_io_test(tmp, erc, epochs, read_erc, comment='another_erp')
    with h5py.File(tmp + '/test.hdf5') as fid:
        assert_true(fid['nice/data/epochs'].keys() != [])


def test_komplexity():
    """Test computation of komplexity measure"""
    epochs = _get_data()[:2]
    komp = KolmogorovComplexity()
    _base_io_test(komp, epochs, read_komplexity)


def test_pe():
    """Test computation of permutation entropy measure"""
    epochs = _get_data()[:2]
    pe = PermutationEntropy()
    _base_io_test(pe, epochs, read_pe)


def test_wsmi():
    """Test computation of wsmi measure"""
    epochs = _get_data()[:2]
    wsmi = SymbolicMutualInformation()
    _base_io_test(wsmi, epochs, read_wsmi)

if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
