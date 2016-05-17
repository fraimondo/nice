# License: BSD (3-clause)

import os.path as op

from nose.tools import assert_equal, assert_true

from numpy.testing import assert_array_equal
import numpy as np
import warnings
import matplotlib

import functools

import mne
import h5py
from mne.utils import _TempDir, clean_warning_registry

# our imports
from nice.measures import PowerSpectralDensity, read_psd
from nice.measures import ContingentNegativeVariation, read_cnv
from nice.measures import KolmogorovComplexity, read_komplexity
from nice.measures import PermutationEntropy, read_pe
from nice.measures import SymbolicMutualInformation, read_smi

from nice.measures import TimeLockedTopography, read_ert

from nice.measures import TimeLockedContrast, read_erc

from nice.measures import WindowDecoding, read_wd
from nice.measures import TimeDecoding, read_td
from nice.measures import GeneralizationDecoding, read_gd
from nice.measures import PowerSpectralDensityEstimator, read_psd_estimator

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


def _get_decoding_data():
    raw = mne.io.Raw(raw_fname, add_eeg_ref=False, proj=False)
    events = mne.read_events(event_name)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True,
                           ecg=True, eog=True, include=['STI 014'],
                           exclude='bads')[::15]

    epochs = mne.Epochs(raw, events, event_id_2, tmin, tmax, picks=picks,
                        preload=preload, decim=3)
    return epochs

clean_warning_registry()  # really clean warning stack


def _compare_values(v, v2):
    if isinstance(v, np.ndarray):
        assert_array_equal(v, v2)
    elif isinstance(v, mne.io.meas_info.Info):
        pass
    elif isinstance(v, dict):
        for key, value in v.items():
            _compare_values(v[key], v2[key])
    elif isinstance(v, PowerSpectralDensityEstimator):
        _compare_instance(v, v2)
    else:
        assert_equal(v, v2)


def _compare_instance(inst1, inst2):
    for k, v in vars(inst1).items():
        v2 = getattr(inst2, k)
        if k == 'ch_info_' and v2 is None:
            continue
        _compare_values(v, v2)


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


def _base_compression_test(inst, epochs):
    orig_shape = inst.data_.shape
    inst.compress(np.mean)
    axis = inst._axis_map['epochs']
    new_shape = np.array(orig_shape)
    new_shape[axis] = 1
    assert_array_equal(inst.data_.shape, new_shape)


def test_spectral():
    """Test computation of spectral measures"""
    epochs = _get_data()[:2]
    psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto',
                       nperseg=128)
    estimator = PowerSpectralDensityEstimator(
        tmin=None, tmax=None, fmin=1., fmax=45., psd_method='welch',
        psd_params=psds_params, comment='default'
    )
    psd = PowerSpectralDensity(estimator, fmin=1., fmax=4.)
    _base_io_test(psd, epochs,
        functools.partial(read_psd, estimators={'default': estimator}))
    # TODO: Fix this test
    # _base_reduction_test(psd, epochs)
    # _base_compression_test(psd, epochs)


def test_time_locked():
    """Test computation of time locked measures"""

    raw = mne.io.Raw(raw_fname, add_eeg_ref=False, proj=False)
    events = mne.read_events(event_name)
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True,
                           ecg=True, eog=True, include=['STI 014'],
                           exclude='bads')[::15]

    epochs = mne.Epochs(raw, events, event_id_2, tmin, tmax, picks=picks,
                        preload=preload, decim=3)
    cnv = ContingentNegativeVariation()
    _base_io_test(cnv, epochs, read_cnv)
    _base_reduction_test(cnv, epochs)

    tmp = _TempDir()
    with h5py.File(tmp + '/test.hdf5') as fid:
        assert_true('nice/data/epochs' not in fid)
    ert = TimeLockedTopography(tmin=0.1, tmax=0.2)
    _erfp_io_test(tmp, ert, epochs, read_ert)
    with h5py.File(tmp + '/test.hdf5') as fid:
        assert_true(fid['nice/data/epochs'].keys() != [])

    tmp = _TempDir()
    with h5py.File(tmp + '/test.hdf5') as fid:
        assert_true('nice/data/epochs' not in fid)
    erc = TimeLockedContrast(tmin=0.1, tmax=0.2, condition_a='a',
                             condition_b='b')
    _erfp_io_test(tmp, erc, epochs, read_erc)
    with h5py.File(tmp + '/test.hdf5') as fid:
        assert_true('nice/data/epochs' in fid)
    erc = TimeLockedContrast(tmin=0.1, tmax=0.2, condition_a='a',
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
    _base_compression_test(komp, epochs)


def test_pe():
    """Test computation of permutation entropy measure"""
    epochs = _get_data()[:2]
    pe = PermutationEntropy()
    _base_io_test(pe, epochs, read_pe)
    _base_reduction_test(pe, epochs)
    _base_compression_test(pe, epochs)


def test_wsmi():
    """Test computation of wsmi measure"""
    epochs = _get_data()[:2]
    method_params = {'bypass_csd': True}
    wsmi = SymbolicMutualInformation(method_params=method_params)
    _base_io_test(wsmi, epochs, read_smi)
    _base_reduction_test(wsmi, epochs)
    _base_compression_test(wsmi, epochs)


def test_window_decoding():
    """Test computation of window decoding"""
    epochs = _get_decoding_data()
    decoding_params = dict(
        sample_weight='auto',
        clf=None,
        cv=None,
        n_jobs=1,
        random_state=42,
        labels=None
    )

    wd = WindowDecoding(tmin=0.1, tmax=0.2, condition_a='a',
                        condition_b='b', decoding_params=decoding_params)
    _base_io_test(wd, epochs, read_wd)


def test_time_decoding():
    """Test computation of time decoding"""
    epochs = _get_decoding_data()
    decoding_params = dict(
        clf=None,
        cv=2,
        n_jobs=1
    )

    td = TimeDecoding(tmin=0.1, tmax=0.2, condition_a='a',
                      condition_b='b', decoding_params=decoding_params)
    _base_io_test(td, epochs, read_td)


def test_generalization_decoding():
    """Test computation of time decoding"""
    epochs = _get_decoding_data()
    decoding_params = dict(
        clf=None,
        cv=2,
        n_jobs=1
    )

    gd = GeneralizationDecoding(tmin=0.1, tmax=0.2, condition_a='a',
                                condition_b='b',
                                decoding_params=decoding_params)
    _base_io_test(gd, epochs, read_gd)

if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
