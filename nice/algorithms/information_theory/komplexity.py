from copy import deepcopy
import time
import numpy as np
import zlib

from mne import pick_info, pick_types
from mne.epochs import EpochsArray
from mne.utils import logger, verbose

@verbose
def epochs_compute_komplexity(epochs, nbins, backend='python',
                              method_params=None, verbose=None):
    """Compute complexity (K)

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs on which to compute the wSMI.
    nbins : int
        Number of bins to use for symbolic transformation
    method_params : dictionary.
        Overrides default parameters for the backend used.
        OpenMP specific {'nthreads'}
    backend : {'python', 'openmp'}
        The backend to be used. Defaults to 'python'.
    """
    picks = pick_types(epochs.info, meg=True, eeg=True)

    if method_params is None:
        method_params = {}

    data = epochs.get_data()[:, picks if picks is not None else Ellipsis]
    logger.info("Running Komplexity")

    if backend == 'python':
        start_time = time.time()
        komp = _komplexity_python(data, nbins)
        elapsed_time = time.time() - start_time
        logger.info("Elapsed time {} sec".format(elapsed_time))
    elif backend == 'openmp':
        from .ompk import komplexity as _ompk_k
        nthreads = (method_params['nthreads']
                    if 'nthreads' in method_params else 1)
        start_time = time.time()
        komp = _ompk_k(data, nbins, nthreads)
        elapsed_time = time.time() - start_time
        logger.info("Elapsed time {} sec".format(elapsed_time))
    else:
        raise ValueError('backend %s not supported for Komplexity'
                         % backend)

    picked_info = pick_info(epochs.info, picks)
    ev_copy = epochs.events.copy()
    evid_copy = deepcopy(epochs.event_id)
    komp = EpochsArray(komp[None].T, picked_info, ev_copy, tmin=0,
                       event_id=evid_copy)
    return komp


def _symb_python(signal, nbins):
    """Compute symbolic transform"""
    ssignal = np.sort(signal)
    items = signal.shape[0]
    first = items / 10
    last = items - first if first > 1 else items - 1
    lower = ssignal[first]
    upper = ssignal[last]
    bsize = (upper - lower) / nbins

    osignal = np.zeros(signal.shape, dtype=np.uint8)
    maxbin = nbins - 1

    for i in range(items):
        tbin = int((signal[i] - lower) / bsize)
        osignal[i] = ((0 if tbin < 0 else maxbin if tbin > maxbin else tbin)
                      + ord('A'))

    return osignal.tostring()


def _komplexity_python(data, nbins):
    """Compute komplexity (K)"""
    ntrials, nchannels, nsamples = data.shape
    k = np.zeros((nchannels, ntrials), dtype=np.float64)
    for trial in range(ntrials):
        for channel in range(nchannels):
            string = _symb_python(data[trial, channel, :], nbins)
            cstring = zlib.compress(string)
            k[channel, trial] = float(len(cstring)) / float(len(string))

    return k
