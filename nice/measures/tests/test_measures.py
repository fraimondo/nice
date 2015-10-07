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
    inst.save(tmp + '/test.hdf5', overwrite='update')
    inst2 = read_fun(tmp + '/test.hdf5')
    _compare_instance(inst, inst2)


def _erfp_io_test(tmp, inst, epochs, read_fun, comment='default'):
    inst.fit(epochs)
    inst.save(tmp + '/test.hdf5', overwrite='update')
    inst2 = read_fun(tmp + '/test.hdf5', epochs, comment=comment)
    assert_array_equal(inst.epochs_.get_data(), inst2.epochs_.get_data())
    _compare_instance(inst, inst2)


def _base_reduction_test(inst, epochs):
    sc = inst.reduce_to_scalar(None)
    if inst.data_.ndim == 3:
        sc2 = np.mean(np.mean(np.mean(inst.data_, axis=0), axis=0), axis=0)
    else:
        sc2 = np.mean(np.mean(inst.data_, axis=0), axis=0)
    assert_equal(sc, sc2)
    topo = inst.reduce_to_topo(None)
    topo_chans = len(mne.io.pick.pick_types(epochs.info, meg=True, eeg=True))
    assert_equal(topo.shape, (topo_chans,))


def test_spectral():
    """Test computation of spectral measures"""
    epochs = _get_data()[:2]
    psd = PowerSpectralDensity(fmin=1, fmax=4)
    _base_io_test(psd, epochs, read_psd)
    _base_reduction_test(psd, epochs)


def test_time_locked():
    """Test computation of time locked measures"""

    epochs = _get_data()[:2]
    cnv = ContingentNegativeVariation()
    _base_io_test(cnv, epochs, read_cnv)
    _base_reduction_test(cnv, epochs)

    tmp = _TempDir()
    with h5py.File(tmp + '/test.hdf5') as fid:
        assert_true('nice/data/epochs' not in fid)
    ert = EventRelatedTopography(tmin=0.1, tmax=0.2)
    _erfp_io_test(tmp, ert, epochs, read_ert)
    with h5py.File(tmp + '/test.hdf5') as fid:
        assert_true(fid['nice/data/epochs'].keys() != [])

    tmp = _TempDir()
    with h5py.File(tmp + '/test.hdf5') as fid:
        assert_true('nice/data/epochs' not in fid)
    erc = EventRelatedContrast(tmin=0.1, tmax=0.2, condition_a='a',
                               condition_b='b')
    _erfp_io_test(tmp, erc, epochs, read_erc)
    with h5py.File(tmp + '/test.hdf5') as fid:
        assert_true('nice/data/epochs' in fid)
    erc = EventRelatedContrast(tmin=0.1, tmax=0.2, condition_a='a',
                               condition_b='b', comment='another_erp')
    _erfp_io_test(tmp, erc, epochs, read_erc, comment='another_erp')
    with h5py.File(tmp + '/test.hdf5') as fid:
        assert_true(fid['nice/data/epochs'].keys() != [])


def test_komplexity():
    """Test computation of komplexity measure"""
    epochs = _get_data()[:2]
    komp = KolmogorovComplexity()
    _base_io_test(komp, epochs, read_komplexity)
    _base_reduction_test(komp, epochs)


def test_pe():
    """Test computation of permutation entropy measure"""
    epochs = _get_data()[:2]
    pe = PermutationEntropy()
    _base_io_test(pe, epochs, read_pe)
    _base_reduction_test(pe, epochs)


def test_wsmi():
    """Test computation of wsmi measure"""
    epochs = _get_data()[:2]
    wsmi = SymbolicMutualInformation()
    _base_io_test(wsmi, epochs, read_wsmi)
    _base_reduction_test(wsmi, epochs)


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
