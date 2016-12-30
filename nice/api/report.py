import mne

from ..modules import _get_module_func


def create_report(instance, title='default', config='default', report=None,
                  config_params=None):
    out = None
    if report is None:
        report = mne.report.Report(title=title)
    func = _get_module_func('report', config)
    out = func(instance, report=report, config_params=config_params)
    return out
