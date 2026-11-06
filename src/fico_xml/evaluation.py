"""
Baselines and model comparison.

The models of the original analysis (:func:`classifiers`) are compared with
stratified 10-fold cross-validation without shuffling (the folds of
the original analysis);
the results are returned as tables.
"""
import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.svm
import sklearn.tree
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

__all__ = ['classifiers', 'majority_baseline', 'threshold_baseline',
           'best_threshold', 'cross_validate', 'compare']

POSITIVE = 'Good'


def classifiers(random_state=0):
    """The models compared in the original analysis (with fixed seeds)."""
    return {
        'decision tree': sklearn.tree.DecisionTreeClassifier(
            min_samples_split=120, random_state=random_state),
        'logistic regression': sklearn.linear_model.LogisticRegression(
            max_iter=1000),
        'perceptron': sklearn.linear_model.Perceptron(
            max_iter=1000, alpha=0.00001, random_state=random_state),
        'SVM (C=0.5)': sklearn.svm.SVC(C=0.5),
        'SVM (C=0.1)': sklearn.svm.SVC(C=0.1),
        'k-NN (k=30)': sklearn.neighbors.KNeighborsClassifier(n_neighbors=30),
        'Gaussian naive Bayes': sklearn.naive_bayes.GaussianNB(),
        'Bernoulli naive Bayes': sklearn.naive_bayes.BernoulliNB(),
        'random forest (min_samples_split=175)':
            sklearn.ensemble.RandomForestClassifier(
                n_estimators=5, min_samples_split=175,
                random_state=random_state),
        'random forest (max_depth=4)':
            sklearn.ensemble.RandomForestClassifier(
                n_estimators=5, max_depth=4, random_state=random_state)}


def _scores(y, y_hat):
    return {'accuracy': accuracy_score(y, y_hat),
            'F1 (Good)': f1_score(y, y_hat, pos_label=POSITIVE)}


def majority_baseline(y):
    """Scores of always predicting the most common class."""
    y = np.asarray(y)
    values, counts = np.unique(y, return_counts=True)
    return _scores(y, np.full(y.shape, values[np.argmax(counts)]))


def threshold_baseline(x, y, threshold):
    """Scores of predicting ``Good`` when ``x >= threshold`` (e.g., the
    external risk estimate)."""
    y_hat = np.where(np.asarray(x) < threshold, 'Bad', POSITIVE)
    return _scores(np.asarray(y), y_hat)


def best_threshold(x, y):
    """The threshold of :func:`threshold_baseline` with the best accuracy
    (searched among the observed values)."""
    x = np.asarray(x, dtype=float)
    candidates = np.unique(x[~np.isnan(x)])
    accuracy = [threshold_baseline(x, y, t)['accuracy'] for t in candidates]
    return float(candidates[int(np.argmax(accuracy))])


def cross_validate(classifier, X, y, n_splits=10):
    """
    Stratified k-fold cross-validation (without shuffling).

    Returns
    -------
    pandas.DataFrame
        Training and test accuracy and F1 score of every fold.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    rows = []
    folds = StratifiedKFold(n_splits=n_splits, shuffle=False)
    for fold, (train, test) in enumerate(folds.split(X, y)):
        model = clone(classifier).fit(X[train], y[train])
        train_scores = _scores(y[train], model.predict(X[train]))
        test_scores = _scores(y[test], model.predict(X[test]))
        rows.append({'fold': fold,
                     **{'train ' + k: v for k, v in train_scores.items()},
                     **{'test ' + k: v for k, v in test_scores.items()}})
    return pd.DataFrame(rows).set_index('fold')


def compare(models, X, y, n_splits=10):
    """
    Cross-validates several models.

    Parameters
    ----------
    models : dict
        ``name -> scikit-learn classifier`` (e.g., :func:`classifiers`).

    Returns
    -------
    pandas.DataFrame
        The mean scores of every model, best test accuracy first.
    """
    summary = {name: cross_validate(model, X, y, n_splits).mean()
               for name, model in models.items()}
    return pd.DataFrame(summary).T.sort_values('test accuracy',
                                               ascending=False)
