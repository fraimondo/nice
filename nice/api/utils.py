import sys
from os import path as op
import time
import subprocess
from distutils.version import LooseVersion
import logging

from mne.utils import logger, WrapStdOut, set_log_file


def _get_git_head(path):
    """Aux function to read HEAD from git"""
    if not isinstance(path, str):
        raise ValueError('path must be a string, you passed a {}'.format(
            type(path))
        )
    if not op.exists(path):
        raise ValueError('This path does not exist: {}'.format(path))
    command = ('cd {gitpath}; '
               'git rev-parse --verify HEAD').format(gitpath=path)
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               shell=True)
    proc_stdout = process.communicate()[0].strip()
    del process
    return proc_stdout


def get_versions(sys):
    """Import stuff and get versions if module

    Parameters
    ----------
    sys : module
        The sys module object.

    Returns
    -------
    module_versions : dict
        The module names and corresponding versions.
    """
    module_versions = {}
    for name, module in sys.modules.items():
        if '.' in name:
            continue
        if '_curses' == name:
            continue
        module_version = LooseVersion(getattr(module, '__version__', None))
        module_version = getattr(module_version, 'vstring', None)
        if module_version is None:
            module_version = None
        elif 'git' in module_version:
            git_path = op.dirname(op.realpath(module.__file__))
            head = _get_git_head(git_path)
            module_version += '-HEAD:{}'.format(head)

        module_versions[name] = module_version
    return module_versions


def log_versions():
    versions = get_versions(sys)

    logger.info('===== Lib Versions =====')
    logger.info('Numpy: {}'.format(versions['numpy']))
    logger.info('Scipy: {}'.format(versions['scipy']))
    logger.info('MNE: {}'.format(versions['mne']))
    logger.info('scikit-learn: {}'.format(versions['sklearn']))
    logger.info('nice: {}'.format(versions['nice']))
    # logger.info('nice-jsmf: {}'.format(versions['njsmf']))
    # TODO: Log nice extensions versions
    logger.info('========================')


def get_run_id():
    """Get the run id

    Returns
    -------
    run_id : str
        A hex hash.
    """
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())


def configure_logging():
    """Set format to file logging and add stdout logging
       Log file messages will be: DATE - LEVEL - MESSAGE
    """
    handlers = logger.handlers
    file_output_format = '%(asctime)s %(levelname)s %(message)s'
    date_format = '%d/%m/%Y %H:%M:%S'
    output_format = '%(message)s'
    for h in handlers:
        if not isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            print('Removing handler {}'.format(h))
        else:
            h.setFormatter(logging.Formatter(file_output_format,
                                             datefmt=date_format))
    lh = logging.StreamHandler(WrapStdOut())
    lh.setFormatter(logging.Formatter(output_format))
    logger.addHandler(lh)


def remove_file_logging():
    """Close and remove logging to file"""
    handlers = logger.handlers
    for h in handlers:
        if isinstance(h, logging.FileHandler):
            h.close()
            logger.removeHandler(h)


def parse_params_from_config(config):
    # TODO: better handling here
    params = {}
    if '?' in config:
        try:
            query = config.split('?')[1]
            for param in query.split('&'):
                k, v = param.split('=')
                if v in ['True', 'true', 'False', 'false']:
                    v = v in ['True', 'true']
                elif '.' in v:
                    v = float(v)
                else:
                    v = int(v)
                params[k] = v
        except:
            raise ValueError('Malformed config query {}'.format(config))
    return params, config.split('?')[0]
