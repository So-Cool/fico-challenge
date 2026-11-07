"""
Explanations of the decisions about individual applicants and of the model.

:func:`explain` takes the raw record of an applicant through the whole
pipeline -- cleaning, consistency checks, feature engineering and prediction --
and explains the prediction with two kinds of counterfactuals
(computed by CtreeX):

* **class-contrastive** -- what would have to change for the opposite
  decision (e.g., "the application would be accepted had the number of
  credit accounts in good standing been at least 20, instead of 15"); and
* **class-equivalent** -- which changes would not alter the decision
  ("the application would still be declined had ...").

Every counterfactual records the *closest* change it suggests, whether the
resulting record would pass the consistency checks and whether it moves a
feature against the monotonicity constraints of the data dictionary
(if the model does not obey them, a warning is issued).
Options restrict the explanations to consistent records (``consistent_only``)
and to features an applicant can change (``actionable_only``).

Applicants without a credit record, with an inconsistent record or with a
missing value of a model feature are not explained.
"""
import dataclasses
import math
import warnings

import pandas as pd

from fico_xml.data import FEATURES
from fico_xml.model import ACTIONABLE, MONOTONICITY
from fico_xml.preprocessing import NO_RECORD, RULES, clean

__all__ = ['Explanation', 'Counterfactual', 'MonotonicityWarning', 'explain',
           'describe', 'plot_tree', 'graphviz_tree', 'OUTCOMES']

_MESSAGES = {rule.name: rule.message for rule in RULES}
EXPLAINED, NO_CREDIT_RECORD, INCONSISTENT, MISSING_VALUES = (
    'explained', 'no credit record', 'inconsistent record', 'missing values')
#: How the classes are described to applicants.
OUTCOMES = {'Good': 'accepted', 'Bad': 'declined'}


class MonotonicityWarning(UserWarning):
    """A counterfactual moves a feature against its monotonicity
    constraint."""


@dataclasses.dataclass
class Counterfactual(object):
    """
    A counterfactual (a set of changes of the model features).

    Attributes
    ----------
    distance : int
        The number of features that change.
    changes : dict
        ``{feature: {'label', 'is', 'wannabe'}}`` -- the label of the
        feature, its value and the described ranges it would need.
    values : dict
        The closest change: ``{feature: value}`` -- the value in the ranges
        nearest to the applicant's value.
    consistent : bool
        Whether the record with the closest change passes the consistency
        checks.
    contradicts_monotonicity : list of str
        The features that move against their monotonicity constraint
        (class-contrastive counterfactuals only).
    """
    distance: int
    changes: dict
    values: dict
    consistent: bool = True
    contradicts_monotonicity: list = dataclasses.field(default_factory=list)

    def text(self):
        """The changes in words (see :func:`describe`)."""
        return describe(self.changes)


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
        ``Good`` (accepted) or ``Bad`` (declined).
    values : dict
        The values of the model features (``feature -> value``).
    contrastive : list of Counterfactual
        Changes that reverse the decision, most important first.
    equivalent : list of Counterfactual
        Changes that keep the decision.
    actionable_only : bool
        Whether the class-contrastive counterfactuals are restricted to the
        features an applicant can change.
    """
    status: str
    messages: list
    prediction: object = None
    values: dict = dataclasses.field(default_factory=dict)
    contrastive: list = dataclasses.field(default_factory=list)
    equivalent: list = dataclasses.field(default_factory=list)
    actionable_only: bool = False

    @property
    def explained(self):
        return self.status == EXPLAINED

    def text(self, limit=3):
        """The explanation in plain English (``limit`` statements of each
        kind)."""
        if not self.explained:
            return ' '.join(self.messages)
        outcome = OUTCOMES[self.prediction]
        other = OUTCOMES['Good' if self.prediction == 'Bad' else 'Bad']
        lines = ['The application is {}.'.format(outcome)]
        if not self.contrastive:
            lines.append('No {}change of the record would make the '
                         'application {}.'.format(
                             'possible ' if self.actionable_only else '',
                             other))
        for prefix, items in (
                ('The application would be {} had '.format(other),
                 self.contrastive),
                ('The application would still be {} had '.format(outcome),
                 self.equivalent)):
            for cf in items[:limit]:
                line = prefix + cf.text() + '.'
                if cf.contradicts_monotonicity:
                    line += (' (This contradicts the expected effect of the '
                             '{}.)').format(' and the '.join(
                                 cf.changes[f]['label']
                                 for f in cf.contradicts_monotonicity))
                lines.append(line)
        return '\n'.join(lines)


def describe(changes):
    """
    Describes changes in words, e.g., "the number of credit checks in the
    last 6 months been at most 2 (instead of 4)" or "there been no credit
    cards with high utilisation (instead of 1)".

    Parameters
    ----------
    changes : dict
        ``{feature: {'label': ..., 'is': value, 'wannabe': [ranges]}}``.
    """
    parts = []
    for change in changes.values():
        label = change['label']
        if change['wannabe'] == ['at most 0'] and label.startswith(
                'number of '):
            parts.append('there been no {} (instead of {})'.format(
                label[len('number of '):], change['is']))
        else:
            parts.append('the {} been {} (instead of {})'.format(
                label, ' or '.join(change['wannabe']), change['is']))
    return ' and '.join(parts)


def _integer_bounds(interval):
    """The whole numbers ``[low, high]`` of an interval (counts: >= 0)."""
    lower, upper = interval[0], interval[1]
    low = 0 if lower is None else max(0, math.floor(float(lower)) + 1)
    high = math.inf if upper is None else math.floor(float(upper))
    return low, high


def _closest(value, intervals, direction=None):
    """The value in the intervals closest to ``value`` (``None`` if no value
    moves in the required ``direction``)."""
    best = None
    for interval in intervals:
        low, high = _integer_bounds(interval)
        if direction == 'decrease':
            high = min(high, value - 1)
        if low > high:
            continue
        candidate = min(max(value, low), high)
        if best is None or abs(candidate - value) < abs(best - value):
            best = candidate
    return best


def _raw_record(raw, values, new):
    """The raw record with the model features changed (the number of credit
    checks in the last 7 days is the difference of two raw features)."""
    record = raw.copy()
    for feature, value in new.items():
        if feature in record.index:
            record[feature] = value
    if 'NumInqLast7DCalc' in new or 'NumInqLast6Mexcl7days' in new:
        excluding = new.get('NumInqLast6Mexcl7days',
                            values['NumInqLast6Mexcl7days'])
        last_7_days = new.get('NumInqLast7DCalc', values['NumInqLast7DCalc'])
        record['NumInqLast6M'] = excluding + last_7_days
    return record


def _counterfactuals(model, x, values, raw, counterfactual, actionable_only):
    included = list(ACTIONABLE) if actionable_only else ()
    explanation = model.explainer.explain_instance(
        x, instance_values=values, counterfactual=counterfactual,
        included=[f for f in included if f in model.features])
    labels = {f: model.explainer.spec.features[f].display_name
              for f in model.features}
    out = []
    for distance, change in explanation.ranked():
        if not change:
            continue
        closest, described, keep = {}, {}, True
        for feature, cf in change.items():
            intervals = cf['wannabe']
            direction = ACTIONABLE.get(feature) if actionable_only else None
            if direction == 'decrease':  # only the part below the value
                intervals = [[lo, hi if hi is not None and math.floor(
                    float(hi)) < values[feature] else str(values[feature] -
                                                          0.5)]
                             for lo, hi in intervals
                             if _integer_bounds([lo, hi])[0] < values[feature]]
            value = _closest(values[feature], intervals, direction)
            if value is None:
                keep = False
                break
            closest[feature] = value
            described[feature] = {
                'label': labels[feature], 'is': values[feature],
                'wannabe': model.explainer.describe(
                    {0: {feature: {'is': [values[feature]],
                                   'wannabe': intervals}}},
                    integer_ranges=True)[0][labels[feature]]['wannabe']}
        if not keep:
            continue
        contradicts = []
        if counterfactual:
            # towards Good the risk has to fall, towards Bad it has to rise
            towards = -1 if explanation.prediction == 'Bad' else 1
            for feature, value in closest.items():
                sign = MONOTONICITY.get(feature, 0)
                step = value - values[feature]
                if sign and step and (step > 0) != (sign * towards > 0):
                    contradicts.append(feature)
        out.append(Counterfactual(distance, described, closest,
                                  contradicts_monotonicity=contradicts))
    if out:  # the consistency of the closest changes (in one batch)
        records = pd.DataFrame([_raw_record(raw, values, cf.values)
                                for cf in out]).reset_index(drop=True)
        removed = clean(records.apply(pd.to_numeric)).removed
        for i, cf in enumerate(out):
            cf.consistent = i not in removed.index
    return explanation.prediction, out


def explain(model, applicant, consistent_only=False, actionable_only=False,
            warn=True):
    """
    Explains the decision about an applicant.

    Parameters
    ----------
    model : fico_xml.model.Model
    applicant : pandas.Series, dict or one-row pandas.DataFrame
        The raw record of the applicant (the 23 features of the data;
        special values such as -9 included).
    consistent_only : bool
        Keep only the counterfactuals whose closest change gives a record
        that passes the consistency checks.
    actionable_only : bool
        Class-contrastive counterfactuals only change the features an
        applicant can act on, in the possible direction
        (:data:`fico_xml.model.ACTIONABLE`).
    warn : bool
        Warn (:class:`MonotonicityWarning`) when class-contrastive
        counterfactuals move a feature against its monotonicity constraint.

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
    record = raw.iloc[0]
    prediction, contrastive = _counterfactuals(model, x, values, record,
                                               True, actionable_only)
    _, equivalent = _counterfactuals(model, x, values, record, False, False)
    if consistent_only:
        contrastive = [cf for cf in contrastive if cf.consistent]
        equivalent = [cf for cf in equivalent if cf.consistent]
    contradicting = sum(bool(cf.contradicts_monotonicity)
                        for cf in contrastive)
    if warn and contradicting:
        warnings.warn('{} of the {} class-contrastive counterfactuals move a '
                      'feature against its monotonicity constraint (the '
                      'model does not obey the constraints of the data '
                      'dictionary).'.format(contradicting, len(contrastive)),
                      MonotonicityWarning, stacklevel=2)
    return Explanation(EXPLAINED, [], prediction, values, contrastive,
                       equivalent, actionable_only=actionable_only)


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
