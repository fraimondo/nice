from math import ceil
import numbers
import warnings

import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold, LabelKFold
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.base import clone
# from sklearn.utils import check_random_state

from mne.parallel import parallel_func
from mne.utils import logger


# def get_roc(x, y, class_a, class_b):
#     fpr, tpr, _ = roc_curve(y, x, pos_label=class_b)
#     auc_score = auc(fpr, tpr)
#     return auc_score


def _decode_window_one_fold(clf, X, y, train, test, sample_weight):
    """Helper function"""
    # XXX: If clf is not SVM, this might fail
    clf.fit(X[train], y[train], svc__sample_weight=sample_weight[train])
    this_probas = clf.predict_proba(X[test])  # use the first column
    prediction = clf.predict(X[test])
    score = roc_auc_score(y_true=y[test], y_score=prediction,
                          sample_weight=sample_weight[test],
                          average='weighted')
    return this_probas, prediction, score


def decode_window(X, y, clf=None, cv=None, sample_weight='auto', n_jobs='auto',
                  random_state=None, labels=None):
    """Decode entire window

    Parameters
    ----------
    X : np.ndarray of float, shape(n_samples, n_sensors, n_times)
        The data.
    y : np.ndarray of int, shape(n_samples,)
        The response vector.
    clf : instance of BaseEstimator | None
        The classifier. If None, defaults to a Pipeline.
    cv : cross validation object | None
        The cross validation. If None, defaults to stratified K-folds
        with 10 folds.
    sample_weight : np.ndarray of float, shape(n_samples,)
        The sample weights to deal with class imbalance.
        if 'auto' computes sample weights to balance

    Returns
    -------
    probas : np.ndarray of float, shape(n_samples,)
        The predicted probabilities for each sample.
    predictions : np.ndarray of int, shape(n_samples,)
        The class preditions.
    scores : np.ndarray of float, shape(n_resamples,)


        The score at each resampling iteration.
    """
    if n_jobs == 'auto':
        try:
            import multiprocessing as mp
            n_jobs = mp.cpu_count()
            logger.info(
                'Autodetected number of jobs {}'.format(n_jobs))
        except:
            logger.info('Cannot autodetect number of jobs')
            n_jobs = 1
    if clf is None:
        scaler = StandardScaler()
        svc = SVC(C=1, kernel='linear', probability=True)
        transform = SelectPercentile(f_classif, 10)
        clf = Pipeline([('scaler', scaler), ('anova', transform), ('svc', svc)])
    if cv is None:
        if labels is None:
            cv = StratifiedKFold(n_folds=int(min(10, len(y) / 2)), y=y,
                                 shuffle=True, random_state=random_state)
        else:
            cv = LabelKFold(labels, n_folds=10)

    if isinstance(sample_weight, str) and sample_weight == 'auto':
        sample_weight = np.zeros(len(y), dtype=float)
        for this_y in np.unique(y):
            this_mask = (y == this_y)
            sample_weight[this_mask] = 1.0 / np.sum(this_mask)

    y = LabelEncoder().fit_transform(y)
    X = X.reshape(len(X), np.prod(X.shape[1:]))
    probas = np.zeros(y.shape, dtype=float)
    predictions = np.zeros(y.shape, dtype=int)
    scores = list()
    parallel, pfunc, _ = parallel_func(_decode_window_one_fold, n_jobs)

    out = parallel(pfunc(clone(clf), X, y, train, test, sample_weight)
                   for train, test in cv)

    for (_, test), (probas_, predicts_, score_) in zip(cv, out):
        probas[test] = probas_[:, 1]  # second column
        predictions[test] = predicts_
        scores.append(score_)

    return probas, predictions, np.array(scores)


def compute_sample_weights(y, class_a, class_b):
    class1 = (y == class_a)
    class2 = (y == class_b)
    sample_weights = np.zeros(len(y), dtype=float)
    sample_weights[class1] = 1. / np.sum(class1)
    sample_weights[class2] = 1. / np.sum(class2)
    return sample_weights
