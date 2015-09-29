from ..externals.h5io import write_hdf5, read_hdf5


class BaseMeasure(object):
    """Base class for M/EEG measures"""

    def save(self, fname):
        write_hdf5(
            fname,
            vars(self),
            title=_get_title(self.__class__, self.comment))

    def fit(self):
        pass

    def transform(self):
        pass


class BaseAverage(BaseMeasure):
    pass


class BaseSpectral(BaseMeasure):
    pass


class BaseConnectivity(BaseMeasure):
    pass


def _get_title(klass, comment):
    if 'measure' in klass.__module__:
        kind = 'measure'
    else:
        raise NotImplementedError('Oh no-- what is this?')

    return '/'.join([
        kind, klass.__name__, comment])


def _read_measure(klass, fname, comment='default'):
    data = read_hdf5(
        fname,  _get_title(klass, comment))
    out = klass(**{k: v for k, v in data.items() if not k.endswith('_')})
    for k, v in data.items():
        if k.endswith('_'):
            setattr(out, k, v)
    return out
