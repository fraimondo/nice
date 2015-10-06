import numpy as np
import mne
from mne.utils import logger, verbose
from scipy.signal import butter, filtfilt


def _get_weights_matrix(nsym):
    """Aux function"""
    wts = np.ones((nsym, nsym))
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    return wts


@verbose
def epochs_compute_wsmi(epochs, kernel, tau, backend='python',
                        method_params=None, verbose=None):
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

    freq = epochs.info['sfreq']

    picks = mne.io.pick.pick_types(epochs.info, meg=True, eeg=True)

    data = epochs.get_data()[:, picks, ...]
    n_epochs = len(data)

    filter_freq = np.double(freq) / kernel / tau
    logger.info('Filtering  at %.2f Hz' % filter_freq)
    b, a = butter(6, 2.0 * filter_freq / np.double(freq), 'lowpass')
    data = np.hstack(data[:, Ellipsis])

    fdata = np.transpose(np.array(
        np.split(filtfilt(b, a, data), n_epochs, axis=1)), [1, 2, 0])
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
            for channel2 in range(channel1+1, nchannels):
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
