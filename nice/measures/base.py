
class BaseMeasure(object):
    """Base class for M/EEG measures"""

    def save(self):
        pass

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
