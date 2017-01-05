import numpy as np
from numpy.polynomial.legendre import legval
from scipy.linalg import inv

from mne import pick_types, pick_info
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.parallel import parallel_func
from mne.channels import read_montage


def _extract_positions(inst, picks):
    """Aux function to get positions via Montage
    """
    montage = read_montage('EGI_256')
    # XXX: EXPERIMENTS -> CSD data shoudl be picked by name
    return montage.pos[picks if picks is not None else Ellipsis]


def _calc_g(cosang, stiffnes=4, num_lterms=50):
    """Calculate spherical spline g function between points on a sphere.

    Parameters
    ----------
    cosang : array-like | float
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffnes : float
        stiffnes of the spline.
    num_lterms : int
        number of Legendre terms to evaluate.
    """
    factors = [(2 * n + 1) / (n ** stiffnes * (n + 1) ** stiffnes * 4 * np.pi)
               for n in range(1, num_lterms + 1)]
    return legval(cosang, [0] + factors)


def _calc_h(cosang, stiffnes=4, num_lterms=50):
    """Calculate spherical spline h function between points on a sphere.

    Parameters
    ----------
    cosang : array-like | float
        cosine of angles between pairs of points on a spherical surface. This
        is equivalent to the dot product of unit vectors.
    stiffnes : float
        stiffnes of the spline. Also referred to as `m`.
    num_lterms : int
        number of Legendre terms to evaluate.
    """
    factors = [(2 * n + 1) /
               (n ** (stiffnes - 1) * (n + 1) ** (stiffnes - 1) * 4 * np.pi)
               for n in range(1, num_lterms + 1)]
    return legval(cosang, [0] + factors)


def _prepare_G(G, lambda2):
    # regularize if desired
    if lambda2 is None:
        lambda2 = 1e-5

    G.flat[::len(G) + 1] += lambda2
    # compute the CSD
    Gi = inv(G)

    TC = Gi.sum(0)
    sgi = np.sum(TC)  # compute sum total

    return Gi, TC, sgi


def _compute_csd(data, G_precomputed, H, head):
    """compute the CSD"""
    n_channels, n_times = data.shape
    mu = data.mean(0)[None]
    Z = data - mu  # XXX? compute average reference
    X = np.zeros_like(data)
    head **= 2

    Gi, TC, sgi = G_precomputed

    Cp2 = np.dot(Gi, Z)
    c02 = np.sum(Cp2, axis=0) / sgi
    C2 = Cp2 - np.dot(TC[:, None], c02[None, :])
    X = np.dot(C2.T, H).T / head
    # for this_time in range(n_times):
    #     # Gi = n_c x n_c
    #     # Z = n_c
    #     Cp = np.dot(Gi, Z[:, this_time])  # compute preliminary C vector
    #     # Cp = n_c
    #     # Sgi = scalar
    #     c0 = np.sum(Cp) / sgi  # common constant across electrodes
    #     # TC = n_c
    #     # c0 = scalar
    #     C = Cp - np.dot(c0, TC.T)  # compute final C vector
    #     # C = n_c
    #     for this_chan in range(n_channels):  # compute all CSDs ...
    #         import pdb; pdb.set_trace()
    #         # ... and scale to head size
    #         # H = n_c
    #         # head = scalar
    #         X[this_chan, this_time] = np.sum(C * H[this_chan].T) / head
    return X


def epochs_compute_csd(inst, picks=None, g_matrix=None, h_matrix=None,
                       lambda2=1e-5, head=1.0, lookup_table_fname=None,
                       n_jobs=1, copy=True):
    """ Current Source Density (CSD) transformation

    Transormation based on spherical spline surface Laplacian as suggested by
    Perrin et al. (1989, 1990), published in appendix of Kayser J, Tenke CE,
    Clin Neurophysiol 2006;117(2):348-368)

    Implementation of algorithms described by Perrin, Pernier, Bertrand, and
    Echallier in Electroenceph Clin Neurophysiol 1989;72(2):184-187, and
    Corrigenda EEG 02274 in Electroenceph Clin Neurophysiol 1990;76:565.

    Parameters
    ----------
    inst : instance of Epochs or Evoked
        The data to be transformed.
    picks : np.ndarray, shape (n_channels,) | None
        The picks to be used. Defaults to None.
    g_matrix : ndarray, shape (n_channels, n_channels) | None
        The matrix oncluding the channel pairwise g function. If None,
        the g_function will be computed from the data (default).
    h_matrix : ndarray, shape (n_channels, n_channels) | None
        The matrix oncluding the channel pairwise g function. If None,
        the h_function will be computed from the data (default).
    lambda2 : float
        Regularization parameter, produces smoothnes. Defaults to 1e-5.
    head : float
        The head radius (unit sphere). Defaults to 1.
    lookup_table_fname : str | None
        The name of the lookup table. Defaults to None. Note. If not None,
        `g_matrix' and `h_matrix' will be ignored.
    n_jobs : int
        The number of processes to run in parallel. Note. Only used for
        Epochs input. Defaults to 1.
    copy : bool
        Whether to overwrite instance data or create a copy.

    Returns
    -------
    inst_csd : instance of Epochs or Evoked
        The transformed data. Output type will match input type.
    """

    if copy is True:
        out = inst.copy()
    else:
        out = inst
    if picks is None:
        picks = pick_types(inst.info, meg=False, eeg=True, exclude='bads')
    if len(picks) == 0:
        raise ValueError('No EEG channels found.')

    if ((g_matrix is None or h_matrix is None) or
       (lookup_table_fname is not None)):
        pos = _extract_positions(inst, picks=picks)

    G = _calc_g(np.dot(pos, pos.T)) if g_matrix is None else g_matrix
    H = _calc_h(np.dot(pos, pos.T)) if h_matrix is None else h_matrix
    G_precomputed = _prepare_G(G, lambda2)
    if isinstance(out, BaseEpochs):
        parallel, my_csd, _ = parallel_func(_compute_csd, n_jobs)
        data = np.asarray(parallel(my_csd(e[picks],
                                   G_precomputed=G_precomputed,
                                   H=H, head=head) for e in out))
        out.preload = True
        out._data = data
    elif isinstance(out, Evoked):
        out.data = _compute_csd(out.data[picks], G_precomputed=G_precomputed,
                                H=H, head=head)
    pick_info(out.info, picks, copy=False)
    return out
