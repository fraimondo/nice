import numpy as np

import mne
from mne.utils import logger, _time_mask
from scipy.signal import butter, filtfilt
from .spatial import epochs_compute_csd


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
        logger.info('Bypassing CSD')
        csd_epochs = epochs
    else:
        logger.info('Computing CSD')
        csd_epochs = epochs_compute_csd(epochs, n_jobs=n_jobs)

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
