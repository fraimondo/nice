import os.path as op

import numpy as np
from scipy import io as sio

from numpy.testing import assert_array_equal, assert_almost_equal
from nose.tools import assert_raises, assert_equal

from nice.utils import create_mock_data_egi
from nice.algorithms.spatial import epochs_compute_csd

from nice.algorithms.spatial import _calc_g
from nice.algorithms.spatial import _calc_h
from nice.algorithms.spatial import _compute_csd
from nice.algorithms.spatial import _extract_positions
from nice.algorithms.spatial import _prepare_G

import mne

n_epochs = 3
raw = create_mock_data_egi(6, n_epochs*386, stim=True)

triggers = np.arange(50, n_epochs*386, 386)

raw._data[-1].fill(0.0)
raw._data[-1, triggers] = [10] * n_epochs

events = mne.find_events(raw)
event_id = {
    'foo': 10,
}
epochs = mne.Epochs(raw, events, event_id, tmin=-.2, tmax=1.34,
                    preload=True, reject=None, picks=None, add_eeg_ref=False,
                    baseline=(None, 0), verbose=False)
epochs.drop_channels(['STI 014'])
picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=False,
                       stim=False, exclude='bads')

csd_data = sio.loadmat(
    op.join(op.realpath(op.dirname(__file__)),
            '..', '..', 'io', 'tests', 'data', 'test-eeg-csd.mat'))


def test_csd_core():
    """Test G, H and CSD against matlab CSD Toolbox"""
    positions = _extract_positions(epochs, picks)
    cosang = np.dot(positions, positions.T)
    G = _calc_g(cosang)
    assert_almost_equal(G, csd_data['G'], 17)
    H = _calc_h(cosang)
    assert_almost_equal(H, csd_data['H'], 16)
    G_precomputed = _prepare_G(G.copy(), lambda2=1e-5)
    for i in range(n_epochs):
        csd_x = _compute_csd(
            epochs._data[i], G_precomputed=G_precomputed, H=H, head=1.0)
        assert_almost_equal(csd_x, csd_data['X'][i], 16)

    assert_almost_equal(G, csd_data['G'], 17)
    assert_almost_equal(H, csd_data['H'], 16)


def test_compute_csd():
    """Test epochs_compute_csd function"""
    csd_epochs = epochs_compute_csd(epochs)
    assert_almost_equal(csd_epochs._data, csd_data['X'], 16)

    csd_evoked = epochs_compute_csd(epochs.average())
    assert_almost_equal(csd_evoked.data, csd_data['X'].mean(0), 16)
    assert_almost_equal(csd_evoked.data, csd_epochs._data.mean(0), 16)

    csd_epochs = epochs_compute_csd(epochs, picks=picks[:4])
    assert_equal(csd_epochs._data.shape, (n_epochs, 4, 386))
    assert_equal(csd_epochs.ch_names, epochs.ch_names[:4])


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
