"""
The decision tree whose decisions are explained.

The model (see :data:`ORIGINAL_FINGERPRINT`) is a decision tree
(``max_depth=7``, ``min_samples_leaf=100``) trained on the 10 count features of
:data:`MODEL_FEATURES`, using the applicants for whom none of them is missing.
"""
import dataclasses
import hashlib

import numpy as np
import pandas as pd
from ctreex import Feature, FeatureSpec, TreeCounterfactualExplainer
from sklearn.tree import DecisionTreeClassifier

from fico_xml.data import feature_metadata

__all__ = ['MODEL_FEATURES', 'MONOTONICITY', 'ACTIONABLE',
           'ORIGINAL_PARAMETERS', 'ORIGINAL_FINGERPRINT',
           'Model', 'model_data', 'train', 'fingerprint']

#: The model features.
MODEL_FEATURES = ['NumTotalTrades', 'NumSatisfactoryTrades',
                  'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec',
                  'NumTradesOpeninLast12M', 'NumRevolvingTradesWBalance',
                  'NumInstallTradesWBalance',
                  'NumBank2NatlTradesWHighUtilization',
                  'NumInqLast6Mexcl7days', 'NumInqLast7DCalc']
#: The model parameters.
ORIGINAL_PARAMETERS = {'max_depth': 7, 'min_samples_leaf': 100}
#: SHA-256 of the structure of the original tree (see :func:`fingerprint`).
ORIGINAL_FINGERPRINT = \
    'b533568b05caef06dcad8830661156b8bfd5df87e587e1259880bc03d6aa7d5a'
_LABELS = {'NumInqLast7DCalc': 'number of credit checks in the last 7 days'}
_CONSTRAINTS = {'Monotonically Increasing': 1, 'Monotonically Decreasing': -1}


def _monotonicity():
    meta = feature_metadata()['features']
    out = {f: _CONSTRAINTS.get(meta[f]['monotonicity'], 0)
           for f in MODEL_FEATURES if f in meta}
    # engineered from the number of credit checks (monotonically increasing)
    out['NumInqLast7DCalc'] = 1
    return {f: out[f] for f in MODEL_FEATURES}


#: How the risk (probability of ``Bad``) should change with every model
#: feature according to the data dictionary: 1 for increases, -1 for
#: decreases and 0 for no constraint.
MONOTONICITY = _monotonicity()
#: The model features an applicant can change and how:
#: paying off balances and not applying for (or opening) new credit
#: only *decrease* them.
ACTIONABLE = {'NumRevolvingTradesWBalance': 'decrease',
              'NumInstallTradesWBalance': 'decrease',
              'NumBank2NatlTradesWHighUtilization': 'decrease',
              'NumInqLast6Mexcl7days': 'decrease',
              'NumInqLast7DCalc': 'decrease',
              'NumTradesOpeninLast12M': 'decrease'}


def fingerprint(classifier):
    """
    SHA-256 of the structure of a fitted decision tree:
    the features, thresholds and children of its nodes (no data).
    """
    tree = classifier.tree_
    parts = [np.asarray(tree.children_left, dtype=np.int64),
             np.asarray(tree.children_right, dtype=np.int64),
             np.asarray(tree.feature, dtype=np.int64),
             np.round(np.asarray(tree.threshold, dtype=np.float64), 9)]
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.tobytes())
    return digest.hexdigest()


def model_data(clean_data, features=MODEL_FEATURES):
    """
    The training data of a model:
    the given features of the applicants for whom none of them is missing.

    Parameters
    ----------
    clean_data : fico_xml.preprocessing.CleanData

    Returns
    -------
    X : pandas.DataFrame
    y : pandas.Series
    """
    X = clean_data.data[list(features)]
    complete = X.notnull().all(axis=1)
    return X[complete], clean_data.target[complete]


def _feature_spec(features):
    meta = feature_metadata()['features']
    return FeatureSpec(
        [(f, None) for f in features],
        features=[Feature(f, 'numerical',
                          display_name=meta[f]['label'] if f in meta
                          else _LABELS.get(f, f),
                          integer=True)
                  for f in features])


@dataclasses.dataclass
class Model(object):
    """
    A fitted decision tree and its counterfactual explainer.

    Attributes
    ----------
    classifier : sklearn.tree.DecisionTreeClassifier
    features : list of str
        The features (in the order of the classifier's columns).
    explainer : ctreex.TreeCounterfactualExplainer
    """
    classifier: DecisionTreeClassifier
    features: list
    explainer: TreeCounterfactualExplainer

    def predict(self, X):
        """Predicts the class (``Good``/``Bad``) of applicants."""
        return self.classifier.predict(np.asarray(X[self.features],
                                                  dtype=float))

    def feature_importance(self):
        """The (Gini) importance of the features, largest first."""
        return pd.Series(self.classifier.feature_importances_,
                         index=self.features,
                         name='importance').sort_values(ascending=False)


def train(clean_data, features=MODEL_FEATURES, random_state=0, monotone=False,
          **parameters):
    """
    Trains a decision tree.

    Parameters
    ----------
    clean_data : fico_xml.preprocessing.CleanData
    features : list of str
    random_state : int
        Seed of the tree
        (scikit-learn breaks ties between equally good splits at random).
    monotone : bool
        Constrain the tree to respect the monotonicity constraints of the
        data dictionary (:data:`MONOTONICITY`);
        the default is the original (unconstrained) tree.
    **parameters
        Parameters of :class:`sklearn.tree.DecisionTreeClassifier`;
        they override the corresponding :data:`ORIGINAL_PARAMETERS`
        (e.g., ``max_depth=5`` keeps ``min_samples_leaf=100``;
        pass ``min_samples_leaf=1`` to lift that limit too).

    Returns
    -------
    Model
    """
    parameters = dict(ORIGINAL_PARAMETERS, **parameters)
    X, y = model_data(clean_data, features)
    if monotone:
        # scikit-learn constrains the probability of the positive class --
        # i.e., the second class, Good -- which moves opposite to the risk
        parameters = dict(parameters, monotonic_cst=[
            -MONOTONICITY.get(f, 0) for f in features])
    classifier = DecisionTreeClassifier(random_state=random_state,
                                        **parameters)
    classifier.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=object))
    explainer = TreeCounterfactualExplainer(classifier,
                                            _feature_spec(list(features)))
    return Model(classifier, list(features), explainer)
