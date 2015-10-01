# License: BSD (3-clause)

import os.path as op

from nose.tools import assert_equal, assert_true

from numpy.testing import assert_array_equal
import numpy as np
import warnings
import matplotlib

import mne
from mne.utils import _TempDir, clean_warning_registry

# our imports
from nice.measures import PowerSpectralDensity
from nice.measures import ContingentNegativeVariation
from nice.measures import KolmogorovComplexity
from nice.measures import PermutationEntropy
from nice.measures import SymbolicMutualInformation
from nice.measures import EventRelatedTopography
from nice.measures import EventRelatedContrast

from nice import Features, read_features


matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
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
        if k == 'epochs_info_' and v2 is None:
            continue
        if isinstance(v, np.ndarray):
            assert_array_equal(v, v2)
        elif isinstance(v, mne.io.meas_info.Info):
            pass
        else:
            assert_equal(v, v2)


def test_collecting_feature():
    """Test computation of spectral measures"""
    epochs = _get_data()[:2]
    measures = [
        PowerSpectralDensity(1, 4),
        ContingentNegativeVariation(),
        EventRelatedTopography(0.1, 0.2),
        EventRelatedContrast(0.1, 0.2, 'a', 'b'),
        EventRelatedContrast(0.1, 0.3, 'a', 'b', comment='another_erp')
    ]

    features = Features(measures)
    # check states and names
    for name, measure in features.items():
        assert_true(not any(k.endswith('_') for k in vars(measure)))
        assert_equal(name, measure._get_title())

    # check order
    assert_equal(list(features.values()), measures)

    # check fit
    features.fit(epochs)
    for measure in measures:
        assert_true(any(k.endswith('_') for k in vars(measure)))

    tmp = _TempDir()
    tmp_fname = tmp + '/test_features.hdf5'
    features.save(tmp_fname)
    features2 = read_features(tmp_fname)
    for ((k1, v1), (k2, v2)) in zip(features.items(), features2.items()):
        assert_equal(k1, k2)
        assert_equal(
            {k: v for k, v in vars(v1).items() if not k.endswith('_')},
            {k: v for k, v in vars(v2).items() if not k.endswith('_')})


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
