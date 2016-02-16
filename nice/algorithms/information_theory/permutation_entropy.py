import math
import numpy as np
from itertools import permutations
from scipy.signal import butter, filtfilt

import mne
from mne.utils import logger, verbose, _time_mask


@verbose
def epochs_compute_pe(epochs, kernel, tau, tmin=None, tmax=None,
                      backend='python', verbose=None):
    """Compute Permutation Entropy (PE)

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs on which to compute the PE.
    kernel : int
        The number of samples to use to transform to a symbol
    tau : int
        The number of samples left between the ones that defines a symbol.
    backend : {'python', 'c'}
        The backend to be used. Defaults to 'python'.
    """
    freq = epochs.info['sfreq']

    picks = mne.io.pick.pick_types(epochs.info, meg=True, eeg=True)

    data = epochs.get_data()[:, picks, ...]
    n_epochs = len(data)

    data = np.hstack(data)

    filter_freq = np.double(freq) / kernel / tau
    logger.info('Filtering  at %.2f Hz' % filter_freq)
    b, a = butter(6, 2.0 * filter_freq / np.double(freq), 'lowpass')

    fdata = np.transpose(np.array(
        np.split(filtfilt(b, a, data), n_epochs, axis=1)), [1, 2, 0])

    time_mask = _time_mask(epochs.times, tmin, tmax)
    fdata = fdata[:, time_mask, :]

    if backend == 'python':
        logger.info("Performing symbolic transformation")
        sym, count = _symb_python(fdata, kernel, tau)
        pe = np.nan_to_num(-np.nansum(count * np.log2(count), axis=1))
    elif backend == 'c':
        from ..optimizations.jivaro import pe as jpe
        pe, sym = jpe(fdata, kernel, tau)
    else:
        raise ValueError('backend %s not supported for PE'
                         % backend)
    nsym = pe.shape[1]
    pe = pe / np.log(nsym)
    return pe, sym


def _define_symbols(kernel):
    result_dict = dict()
    total_symbols = math.factorial(kernel)
    cursymbol = 0
    for perm in permutations(range(kernel)):
        order = ''.join(map(str, perm))
        if order not in result_dict:
            result_dict[order] = cursymbol
            cursymbol = cursymbol + 1
            result_dict[order[::-1]] = total_symbols - cursymbol
    result = []
    for v in range(total_symbols):
        for symbol, value in result_dict.items():
            if value == v:
                result += [symbol]
    return result


# Performs symbolic transformation accross 1st dimension
def _symb_python(data, kernel, tau):
    """Compute symbolic transform"""
    symbols = _define_symbols(kernel)
    dims = data.shape

    signal_sym_shape = list(dims)
    signal_sym_shape[1] = data.shape[1] - tau * (kernel - 1)
    signal_sym = np.zeros(signal_sym_shape, np.int32)

    count_shape = list(dims)
    count_shape[1] = len(symbols)
    count = np.zeros(count_shape, np.int32)

    for k in range(signal_sym_shape[1]):
        subsamples = range(k, k + kernel * tau, tau)
        ind = np.argsort(data[:, subsamples], 1)
        signal_sym[:, k, ] = np.apply_along_axis(
            lambda x: symbols.index(''.join(map(str, x))), 1, ind)

    count = np.double(np.apply_along_axis(
        lambda x: np.bincount(x, minlength=len(symbols)), 1, signal_sym))

    return signal_sym, (count / signal_sym_shape[1])
