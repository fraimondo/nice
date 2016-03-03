import os.path as op
import numpy as np
from numpy.polynomial.legendre import legval
from scipy.linalg import inv

import mne
from mne.utils import logger, _time_mask
from scipy.signal import butter, filtfilt


def _get_weights_matrix(nsym):
    """Aux function"""
    wts = np.ones((nsym, nsym))
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    return wts


def epochs_compute_wsmi(epochs, kernel, tau, tmin=None, tmax=None,
                        backend='python', method_params=None, n_jobs='auto'):
    """Compute weighted mutual symbolic information (wSMI)

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs on which to compute the wSMI.
    kernel : int
        The number of samples to use to transform to a symbol
    tau : int
        The number of samples left between the ones that defines a symbol.
    method_params : dictionary.
        Overrides default parameters.
        OpenMP specific {'nthreads'}
    backend : {'python', 'openmp'}
        The backend to be used. Defaults to 'pytho'.
    """
    if method_params is None:
        method_params = {}

    if n_jobs == 'auto':
        try:
            import multiprocessing as mp
            import mkl
            n_jobs = int(mp.cpu_count() / mkl.get_max_threads())
            logger.info(
                'Autodetected number of jobs {}'.format(n_jobs))
        except:
            logger.info('Cannot autodetect number of jobs')
            n_jobs = 1

    if 'bypass_csd' in method_params and method_params['bypass_csd'] is True:
        csd_epochs = epochs
    else:
        csd_epochs = compute_csd(epochs, n_jobs=n_jobs)

    freq = csd_epochs.info['sfreq']

    picks = mne.io.pick.pick_types(csd_epochs.info, meg=True, eeg=True)

    data = csd_epochs.get_data()[:, picks, ...]
    n_epochs = len(data)

    filter_freq = np.double(freq) / kernel / tau
    logger.info('Filtering  at %.2f Hz' % filter_freq)
    b, a = butter(6, 2.0 * filter_freq / np.double(freq), 'lowpass')
    data = np.hstack(data)

    fdata = np.transpose(np.array(
        np.split(filtfilt(b, a, data), n_epochs, axis=1)), [1, 2, 0])

    time_mask = _time_mask(epochs.times, tmin, tmax)
    fdata = fdata[:, time_mask, :]
    if backend == 'python':
        from .information_theory.permutation_entropy import _symb_python
        logger.info("Performing symbolic transformation")
        sym, count = _symb_python(fdata, kernel, tau)
        nsym = count.shape[1]
        wts = _get_weights_matrix(nsym)
        logger.info("Running wsmi with python...")
        wsmi, smi = _wsmi_python(sym, count, wts)
    elif backend == 'openmp':
        from .optimizations.jivaro import wsmi as jwsmi
        nsym = np.math.factorial(kernel)
        wts = _get_weights_matrix(nsym)
        nthreads = (method_params['nthreads'] if 'nthreads' in
                    method_params else 1)
        if nthreads == 'auto':
            try:
                import mkl
                nthreads = mkl.get_max_threads()
                logger.info(
                    'Autodetected number of threads {}'.format(nthreads))
            except:
                logger.info('Cannot autodetect number of threads')
                nthreads = 1
        wsmi, smi, sym, count = jwsmi(fdata, kernel, tau, wts, nthreads)
    else:
        raise ValueError('backend %s not supported for wSMI'
                         % backend)

    return wsmi, smi, sym, count


def _wsmi_python(data, count, wts):
    """Compute wsmi"""
    nchannels, nsamples, ntrials = data.shape
    nsymbols = count.shape[1]
    smi = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)
    wsmi = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)
    for trial in range(ntrials):
        for channel1 in range(nchannels):
            for channel2 in range(channel1 + 1, nchannels):
                pxy = np.zeros((nsymbols, nsymbols))
                for sample in range(nsamples):
                    pxy[data[channel1, sample, trial],
                        data[channel2, sample, trial]] += 1
                pxy = pxy / nsamples
                for sc1 in range(nsymbols):
                    for sc2 in range(nsymbols):
                        if pxy[sc1, sc2] > 0:
                            aux = pxy[sc1, sc2] * np.log(
                                pxy[sc1, sc2] / count[channel1, sc1, trial] /
                                count[channel2, sc2, trial])
                            smi[channel1, channel2, trial] += aux
                            wsmi[channel1, channel2, trial] += (wts[sc1, sc2] *
                                                                aux)
    wsmi = wsmi / np.log(nsymbols)
    smi = smi / np.log(nsymbols)
    return wsmi, smi


def _extract_positions(inst, picks):
    """Aux function to get positions via Montage
    """
    fname = None
    if inst.info['description'] in ('geodesic256', 'geodesic257'):
        fname = op.realpath(op.join(op.dirname(__file__), 'templates',
                            'EGI_256.csd'))
    else:
        raise ValueError('CSD Coordinates for {} are not supplied yet. '
                         'Please contact the developers '
                         '(or set the correct description in '
                         'epochs.info[''description'']).'.format(
                             inst.info['description']))
    montage = mne.channels.read_montage(fname)
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
    logger.info('Calculating G')
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
    logger.info('Calculating H')
    factors = [(2 * n + 1) /
               (n ** (stiffnes - 1) * (n + 1) ** (stiffnes - 1) * 4 * np.pi)
               for n in range(1, num_lterms + 1)]
    return legval(cosang, [0] + factors)


def _prepare_G(G, lambda2):
    logger.info('Preparing G')
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

    for this_time in range(n_times):
        Cp = np.dot(Gi, Z[:, this_time])  # compute preliminary C vector
        c0 = np.sum(Cp) / sgi  # common constant across electrodes
        C = Cp - np.dot(c0, TC.T)  # compute final C vector
        for this_chan in range(n_channels):  # compute all CSDs ...
            # ... and scale to head size
            X[this_chan, this_time] = np.sum(C * H[this_chan].T) / head
    return X


def compute_csd(inst, picks=None, g_matrix=None, h_matrix=None,
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
        picks = mne.pick_types(inst.info, meg=False, eeg=True, exclude='bads')
    if len(picks) == 0:
        raise ValueError('No EEG channels found.')

    logger.info('Computing CSD')
    if ((g_matrix is None or h_matrix is None) or
       (lookup_table_fname is not None)):
        pos = _extract_positions(inst, picks=picks)

    G = _calc_g(np.dot(pos, pos.T)) if g_matrix is None else g_matrix
    H = _calc_h(np.dot(pos, pos.T)) if h_matrix is None else h_matrix
    G_precomputed = _prepare_G(G, lambda2)
    logger.info('Applying G and H')
    if isinstance(out, mne.epochs._BaseEpochs):
        parallel, my_csd, _ = mne.parallel.parallel_func(_compute_csd, n_jobs)
        data = np.asarray(parallel(my_csd(e[picks],
                                   G_precomputed=G_precomputed,
                                   H=H, head=head) for e in out))
        out.preload = True
        out._data = data
    elif isinstance(out, mne.evoked.Evoked):
        out.data = _compute_csd(out.data[picks], G_precomputed=G_precomputed,
                                H=H, head=head)
    mne.pick_info(out.info, picks, copy=False)
    logger.info('CSD Done')
    return out
