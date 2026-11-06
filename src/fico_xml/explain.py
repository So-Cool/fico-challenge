"""
Explanations of the decisions about individual applicants and of the model.

:func:`explain` takes the raw record of an applicant through the whole
pipeline -- cleaning, consistency checks, feature engineering and prediction --
and explains the prediction with two kinds of counterfactuals
(computed by CtreeX):

* **class-contrastive** -- what would have to change for the opposite
  decision (e.g., "the application would be Good had the number of credit
  accounts in good standing been at least 20, instead of 15"); and
* **class-equivalent** -- which changes would not alter the decision
  ("the application would still be Bad had ...").

Applicants without a credit record, with an inconsistent record or with a
missing value of a model feature are not explained.
"""
import dataclasses

import pandas as pd

from fico_xml.data import FEATURES
from fico_xml.preprocessing import NO_RECORD, RULES, clean

__all__ = ['Explanation', 'explain', 'describe', 'plot_tree',
           'graphviz_tree']

_MESSAGES = {rule.name: rule.message for rule in RULES}
EXPLAINED, NO_CREDIT_RECORD, INCONSISTENT, MISSING_VALUES = (
    'explained', 'no credit record', 'inconsistent record', 'missing values')


@dataclasses.dataclass
class Explanation(object):
    """
    The explanation of the decision about an applicant.

    Attributes
    ----------
    status : str
        ``'explained'``, ``'no credit record'``, ``'inconsistent record'`` or
        ``'missing values'``.
    messages : list of str
        Why the applicant could not be explained (if so).
    prediction : str or None
        ``Good`` or ``Bad``.
    values : dict
        The values of the model features (``feature -> value``).
    contrastive : list
        ``(distance, {feature: {'is', 'wannabe'}})`` pairs -- changes that
        reverse the decision -- most important first.
    equivalent : list
        The same for changes that keep the decision.
    """
    status: str
    messages: list
    prediction: object = None
    values: dict = dataclasses.field(default_factory=dict)
    contrastive: list = dataclasses.field(default_factory=list)
    equivalent: list = dataclasses.field(default_factory=list)

    @property
    def explained(self):
        return self.status == EXPLAINED

    def text(self, limit=3):
        """The explanation in plain English (``limit`` statements of each
        kind)."""
        if not self.explained:
            return ' '.join(self.messages)
        other = 'Good' if self.prediction == 'Bad' else 'Bad'
        lines = ['The application is assessed as {}.'.format(self.prediction)]
        for prefix, items in (
                ('The application would be {} had '.format(other),
                 self.contrastive),
                ('The application would still be {} had '.format(
                    self.prediction), self.equivalent)):
            for _, change in items[:limit]:
                lines.append(prefix + describe(change) + '.')
        return '\n'.join(lines)


def describe(change):
    """
    Describes a counterfactual in words, e.g., "the number of credit checks
    in the last 6 months been at most 2 (instead of 4)".

    Parameters
    ----------
    change : dict
        ``{label: {'is': [value], 'wannabe': [ranges]}}`` (described).
    """
    parts = []
    for label, cf in change.items():
        parts.append('the {} been {} (instead of {})'.format(
            label, ' or '.join(cf['wannabe']), cf['is'][0]))
    return ' and '.join(parts)


def _counterfactuals(model, x, values, counterfactual):
    explanation = model.explainer.explain_instance(
        x, instance_values=values, counterfactual=counterfactual)
    out = []
    for distance, change in explanation.ranked():
        if change:
            out.append((distance, model.explainer.describe(
                {0: change}, integer_ranges=True)[0]))
    return explanation.prediction, out


def explain(model, applicant):
    """
    Explains the decision about an applicant.

    Parameters
    ----------
    model : fico_xml.model.Model
    applicant : pandas.Series, dict or one-row pandas.DataFrame
        The raw record of the applicant (the 23 features of the data;
        special values such as -9 included).

    Returns
    -------
    Explanation
    """
    if isinstance(applicant, dict):
        applicant = pd.Series(applicant)
    if isinstance(applicant, pd.Series):
        applicant = applicant.to_frame().T
    if len(applicant) != 1:
        raise ValueError('Explain one applicant at a time.')
    raw = applicant[FEATURES].apply(pd.to_numeric)
    result = clean(raw)
    if len(result.removed):
        reason = result.removed.iloc[0]
        if reason == NO_RECORD:
            return Explanation(NO_CREDIT_RECORD, [
                'There is no credit record of this applicant; '
                'the application has to be assessed manually.'])
        violated = result.violations.iloc[0]
        return Explanation(INCONSISTENT, [
            _MESSAGES[name] for name in violated.index[violated]] + [
            'The record has to be corrected before it can be assessed.'])
    row = result.data.iloc[0]
    values = {f: row[f] for f in model.features}
    missing = [f for f, v in values.items() if pd.isnull(v)]
    if missing:
        labels = {f: model.explainer.spec.features[f].display_name
                  for f in missing}
        return Explanation(MISSING_VALUES, [
            'The record has no usable value of the {}; the model cannot '
            'assess it.'.format(' and the '.join(labels.values()))],
            values=values)
    values = {f: int(v) for f, v in values.items()}
    x = [values[f] for f in model.features]
    prediction, contrastive = _counterfactuals(model, x, values, True)
    _, equivalent = _counterfactuals(model, x, values, False)
    return Explanation(EXPLAINED, [], prediction, values, contrastive,
                       equivalent)


def plot_tree(model, ax=None, max_depth=None, fontsize=7):
    """Plots the decision tree (matplotlib)."""
    from sklearn.tree import plot_tree as _plot_tree
    return _plot_tree(model.classifier, feature_names=model.features,
                      class_names=list(model.classifier.classes_),
                      filled=True, rounded=True, impurity=False,
                      proportion=True, max_depth=max_depth, ax=ax,
                      fontsize=fontsize)


def graphviz_tree(model):
    """Renders the decision tree with Graphviz (the ``graphviz`` extra and
    the Graphviz binary are needed)."""
    from ctreex.sklearn_tools import graphviz_tree as _graphviz_tree
    return _graphviz_tree(model.classifier, model.features)
