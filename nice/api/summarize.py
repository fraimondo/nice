import os
import os.path as op
from glob import glob
from copy import deepcopy

import numpy as np
from scipy import io as sio

import pandas as pd

from mne.utils import logger

from . reductions import get_reductions
from .. features import Features, read_features


class Summary(object):

    def __init__(self):
        self._scalar_names = None
        self._scalars = None

        self._topo_names = None
        self._topos = None
        self._topo_subjects = None

    def add_topo(self, names, values, reduction_name):
        if self._topo_names is None:
            self._topo_names = names
        elif names != self._topo_names:
            raise ValueError('Summary topo names do not match')
        if self._topos is None:
            self._topos = {}
        self._topos[reduction_name] = values[..., None]

    def add_scalar(self, names, values, reduction_name):
        if self._scalar_names is None:
            self._scalar_names = names
        elif sorted(names) != sorted(self._scalar_names):
            raise ValueError('Summary scalar names do not match')
        data = dict(zip(names, values))
        data['Reduction'] = reduction_name
        if self._scalars is None:
            self._scalars = pd.DataFrame(data, index=[0])
        else:
            ts = pd.DataFrame(data, index=[len(self._scalars)])
            self._scalars = pd.concat([self._scalars, ts])

    def scalars(self):
        return self._scalars

    def topos(self):
        return self._topos

    def __repr__(self):
        if self._scalar_names is None:
            s_string = 'Empty'
        else:
            s_string = '({}): {}'.format(
                len(self._scalar_names), len(self._scalars))

        if self._topo_names is None:
            t_string = 'Empty'
        else:
            t_string = '({}): {}'.format(
                len(self._topo_names), len(self._topos))

        n_subjects = 1
        if self._topo_subjects is not None:
            n_subjects = len(self._topo_subjects)
        s = '<Summary | Scalars: {} - Topos: {} - Subjects: {} >'
        s = s.format(s_string, t_string, n_subjects)

        return s

    def save(self, prefix):
        self._scalars.to_csv('{}_scalars.csv'.format(prefix), sep=';')
        copy = {}
        if self._topos is not None:
            copy = {k: v for k, v in self._topos.items()}
            copy['names'] = self._topo_names
            if self._topo_subjects is not None:
                copy['subjects'] = self._topo_subjects
        sio.savemat('{}_topos.mat'.format(prefix), copy)

    def copy(self):
        out = Summary()
        if self._scalars is not None:
            out._scalar_names = list(self._scalar_names)
            out._scalars = self._scalars.copy(deep=True)

        if self._topos is not None:
            out._topo_names = list(self._topo_names)
            out._topos = deepcopy(self._topos)
        if self._topo_subjects is not None:
            out._topo_subjects = list(self._topo_subjects)
        return out

    def filter(self, reductions):
        out = self.copy()
        out._scalars = out._scalars[out._scalars['Reduction'].isin(reductions)]
        for k in list(out._topos.keys()):
            if k not in reductions:
                del out._topos[k]
        return out

    def _check_integrity(self):
        # Check data:
        #   - Same Subjects
        #   - Topos shape are correct
        t_subjects = self._topo_subjects
        n_subjects = len(t_subjects) if t_subjects is not None else 1
        if 'Subject' in self._scalars.columns:
            if t_subjects is None and self._topos is not None:
                raise ValueError('Subjects lists do not match')
            else:
                s_subjects = np.unique(self._scalars['Subject'].values)
                if sorted(t_subjects) != sorted(s_subjects):
                    raise ValueError('Subjects list items do not match')
        if self._topos is not None:
            n_topos = len(self._topo_names)
            for k, v in self._topos.items():
                if v.shape[-1] != n_subjects:
                    raise ValueError('Number of subjects do not match topos')
                if v.shape[0] != n_topos:
                    raise ValueError('Number of topos do not match object')

    def _make_back_compat(self):
        # COMPAT
        # Previous single subject topos summaries were 2D
        for k, v in self._topos.items():
            if v.ndim != 3:
                self._topos[k] = v[..., None]

        # This two columns has caps
        if 'reduction' in self._scalars.columns:
            self._scalars.rename(
                columns={'reduction': 'Reduction'}, inplace=True)
        if 'subject' in self._scalars.columns:
            self._scalars.rename(
                columns={'subject': 'Subject'}, inplace=True)
            if len(np.unique(self._scalars['Subject'].values)) == 1:
                del self._scalars['Subject']


def _concatenate_summaries(summaries, names):
    """Concatenate list of summaries using subject from names

    Parameters
    ----------
    summaries : list of summary
        The input summaries.
    names : list of str
        list of subjects names
    Returns
    -------
    out : instance of summary
    """
    results = Summary()
    scalar_reductions = None
    for t_s, t_n in zip(summaries, names):
        t_df = t_s._scalars.copy(deep=True)
        t_df['Subject'] = t_n
        if results._scalars is None:
            results._scalars = t_df
            results._scalar_names = list(t_s._scalar_names)
            scalar_reductions = np.unique(t_s._scalars['Reduction'].values)
        else:
            if (t_s._scalar_names is None or sorted(results._scalar_names) !=
                    sorted(t_s._scalar_names)):
                raise ValueError('Scalars do not have the same measures')
            elif (sorted(np.unique(t_df['Reduction'].values)) !=
                    sorted(scalar_reductions)):
                raise ValueError('Scalars reductions do not match')
            else:
                results._scalars = pd.concat(
                    [results._scalars, t_df])
        if results._topos is None:
            results._topo_names = list(t_s._topo_names)
            results._topos = t_s._topos.copy()
            results._topo_subjects = [t_n]
        else:
            if results._topo_names != t_s._topo_names:
                raise ValueError('Topos do not have the same measures')
            for k, v in results._topos.items():
                results._topos[k] = np.concatenate([v, t_s._topos[k]], axis=-1)
            results._topo_subjects.append(t_n)
    return results


def read_summary(prefix):
    df = pd.read_csv('{}_scalars.csv'.format(prefix), sep=';', index_col=0)
    if len(df.columns) == 0:  # COMPAT
        # Might be old CSV, separated by ','
        df = pd.read_csv('{}_scalars.csv'.format(prefix), sep=',', index_col=0)
    mc = sio.loadmat('{}_topos.mat'.format(prefix))
    suma = Summary()
    suma._scalars = df
    suma._scalar_names = [x for x in df.columns if x.startswith('nice')]
    topo_names = [x.strip() for x in mc['names']]
    topo_subjects = None
    if 'subjects' in mc:
        topo_subjects = [x.strip() for x in mc['subjects']]
    suma._topo_subjects = topo_subjects
    suma._topo_names = topo_names
    k = [x for x in mc.keys()
         if not (x in ['names', 'subjects'] or x.startswith('_'))]
    suma._topos = {}
    for t_k in k:
        suma._topos[t_k] = mc[t_k]
    suma._make_back_compat()  # COMPAT
    suma._check_integrity()
    return suma


def _try_read_summary(path):
    # Try to load previous results
    csv_s = sorted(glob(op.join(path, '*_scalars.csv')))
    if len(csv_s) > 0:
        if len(csv_s) > 1:
            logger.warning('More than one scalars CSV file in {}'
                           ' Using last one'.format(path))
    else:
        return False
    mat_s = sorted(glob(op.join(path, '*_topos.mat')))
    if len(mat_s) > 0:
        if len(mat_s) > 1:
            logger.warning('More than one topo MAT file i {} '
                           ' Using lat one'.format(path))
    s_prefix = csv_s[-1][:-12]
    t_prefix = mat_s[-1][:-10]
    if s_prefix != t_prefix:
        raise ValueError('Different prefix for scalars and topos.'
                         ' Clean database or Recompute')

    logger.info('Reading previous results')
    summary = read_summary(s_prefix)
    return summary


def summarize_subject(features, reductions, out_path=None, recompute=False):
    """Summarizes one subject

    Parameters
    ----------
    features : instance of feature collection or string
        The input data. If string, look for and HDF5 feature collection.
    reductions : list of str
        list of reductions to use
    out_path : str
        path to store the summary object. If None (default),
        results will not be saved.
    recompute : bool
        If true, try to use previously saved results. If false,
        recompute and overwrite. Results will be read from out_path
    Returns
    -------
    out : instance of summary
    """
    reductions_to_do = reductions
    summary = False
    if recompute is False and (out_path is not None or
                               isinstance(features, str)):
        summary = False  # not Found
        if isinstance(features, str):
            logger.info('Trying to get summary from {}'.format(features))
            summary = _try_read_summary(features)
            if summary is False:
                logger.info('Summary not found in features path ')
        if summary is False and out_path is not None:
            logger.info('Trying to get summary from {}'.format(out_path))
            summary = _try_read_summary(out_path)
            if summary is False:
                logger.info('Summary not found in out_path path ')
        if summary is not False:
            s_reductions = np.unique(summary.scalars()['Reduction'].values)
            t_reductions = list(summary.topos().keys())
            if sorted(s_reductions) != sorted(t_reductions):
                raise ValueError('Scalar and topos reductions do not match. '
                                 'Recompute')

            reductions_to_do = [x for x in reductions if x not in s_reductions]
            logger.info('Using previous reductions.')

    if len(reductions_to_do) == 0:
        logger.info('All reductions were done')
        if summary is not False:
            summary = summary.filter(reductions)
        return summary
    elif summary is False:
        summary = Summary()

    out_prefix = 'default'
    if isinstance(features, str):
        fc_files = sorted(glob(op.join(features, '*.hdf5')))
        if len(fc_files) > 1:
            logger.warning('More than one HDF5 file for {}.'
                           ' Using last one'.format(features))
        elif len(fc_files) == 0:
            logger.warning('No HDF5 file for in {}.'.format(features))
            return None
        logger.info('Reading features from {}'.format(fc_files[-1]))
        fc = read_features(fc_files[-1])
        if fc_files[-1].endswith('_features.hdf5'):
            out_prefix = fc_files[-1][:-14]
        else:
            out_prefix = fc_files[-1][:-5]
    elif isinstance(features, Features):
        fc = features
    else:
        raise ValueError('What are you trying to reduce?')

    logger.info('Proceding with following reductions:')
    for reduction in reductions_to_do:
        logger.info('\t{}'.format(reduction))
    for i_red, reduction_name in enumerate(reductions_to_do):
        logger.info('Applying {}'.format(reduction))
        reduction = get_reductions(reduction_name)
        scalars = fc.reduce_to_scalar(reduction)
        topos = fc.reduce_to_topo(reduction)

        topo_names = fc.topo_names()
        scalar_names = fc.scalar_names()
        summary.add_scalar(scalar_names, scalars, reduction_name)
        summary.add_topo(topo_names, topos, reduction_name)

    if out_path is not None:
        where = op.join(out_path, out_prefix)
        logger.info('Saving summary to {}'.format(where))
        summary.save(where)

    # Filter only reductions that we were actually asked
    out = summary.filter(reductions)
    return out


def summarize_run(in_path, reductions, subject_extra_info=None, out_path=None,
                  recompute=False):
    """Summarizes a folder with several subjects results subfolders

    Parameters
    ----------
    in_path : str
        Path to subjects' results subfolders
    reductions : list of str
        list of reductions to use
    subject_extra_info : pandas.DataFrame object
        If not None (default) it will use the 'Subject' column to get the list
        of subjects to include and it will append all the other columns
        to the summary dataframe
    out_path : str
        path to store the summary object. If None (default),
        results will not be saved.
    recompute : bool
        If true, try to use previously saved results. If false,
        recompute and overwrite. Results will be read from out_path
    Returns
    -------
    out : instance of summary
    """
    if subject_extra_info is not None:
        subjects = list(subject_extra_info['Subject'].values)
    else:
        subjects = [x[0].split('/')[-1] for x in os.walk(in_path)]
        subjects = [x for x in subjects if len(x) > 0]
    logger.info('Summarizing {} subjects'.format(len(subjects)))

    summaries = []
    subjects_done = []
    for subject in subjects:
        sin_path = op.join(in_path, subject)
        summary = summarize_subject(sin_path, reductions,
                                    out_path=out_path, recompute=recompute)
        if summary is not None and summary is not False:
            summaries.append(summary)
            subjects_done.append(subject)
    logger.info('Could summarize {} subjects out of {}'.format(
        len(subjects_done), len(subjects)))
    global_summary = None
    if len(subjects_done) > 0:
        global_summary = _concatenate_summaries(summaries, subjects_done)
        if out_path is not None:
            global_summary.save(op.join(out_path, 'all'))
    return global_summary
