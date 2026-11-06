import numpy as np
import pandas as pd
import pytest

from conftest import consistent_applicant
from fico_xml import evaluation, explain as E, model as M
from fico_xml import preprocessing as pp
from fico_xml.data import FEATURES


def test_model(model, raw):
    assert model.features == M.MODEL_FEATURES
    assert model.classifier.get_depth() <= 7
    importance = model.feature_importance()
    assert set(importance.index) == set(M.MODEL_FEATURES)
    assert importance.is_monotonic_decreasing
    X, y = M.model_data(pp.clean(raw))
    assert set(model.predict(X)) <= {'Good', 'Bad'}
    assert len(M.fingerprint(model.classifier)) == 64


def _apply(values, change, labels):
    """Applies a described change (whole-number ranges) to the values."""
    new = dict(values)
    by_label = {v: k for k, v in labels.items()}
    for label, cf in change.items():
        text = cf['wannabe'][0]
        words = text.split()
        if text.startswith('at most') or text.startswith('exactly'):
            value = int(words[-1])
        elif text.startswith('at least'):
            value = int(words[-1])
        else:  # between a and b
            value = int(words[1])
        new[by_label[label]] = value
    return new


def test_explanations_change_the_prediction(model, raw):
    labels = {f: model.explainer.spec.features[f].display_name
              for f in model.features}
    for i in raw.index[:40]:
        e = E.explain(model, raw.loc[i])
        assert e.explained, e.messages
        frame = pd.DataFrame([e.values])
        assert model.predict(frame)[0] == e.prediction
        for _, change in e.contrastive[:5]:
            new = pd.DataFrame([_apply(e.values, change, labels)])
            assert model.predict(new)[0] != e.prediction
        for _, change in e.equivalent[:5]:
            new = pd.DataFrame([_apply(e.values, change, labels)])
            assert model.predict(new)[0] == e.prediction
        assert e.prediction in e.text()


def test_refusals(model):
    no_record = E.explain(model, {f: -9 for f in FEATURES})
    assert no_record.status == E.NO_CREDIT_RECORD and not no_record.explained
    row = consistent_applicant(np.random.default_rng(3))
    row.update(NumTotalTrades=2, NumSatisfactoryTrades=5,
               NumTradesOpeninLast12M=0, NumRevolvingTradesWBalance=0,
               NumInstallTradesWBalance=0)
    inconsistent = E.explain(model, row)
    assert inconsistent.status == E.INCONSISTENT
    assert 'more credit accounts in good standing' in inconsistent.text()
    row = consistent_applicant(np.random.default_rng(3))
    row['NumRevolvingTradesWBalance'] = -8
    missing = E.explain(model, row)
    assert missing.status == E.MISSING_VALUES
    assert 'revolving accounts' in missing.text()
    with pytest.raises(ValueError):
        E.explain(model, pd.DataFrame([row, row]))


def test_plot_tree(model):
    import matplotlib
    matplotlib.use('Agg')
    assert E.plot_tree(model, max_depth=2)


def test_evaluation(raw):
    X, y = M.model_data(pp.clean(raw))
    assert evaluation.majority_baseline(y)['accuracy'] >= 0.5
    t = evaluation.best_threshold(raw['ExternalRiskEstimate'],
                                  raw['RiskPerformance'])
    assert evaluation.threshold_baseline(raw['ExternalRiskEstimate'],
                                         raw['RiskPerformance'], t)
    folds = evaluation.cross_validate(
        evaluation.classifiers()['decision tree'], X, y, n_splits=3)
    assert len(folds) == 3 and (folds['test accuracy'] > 0.5).all()
    table = evaluation.compare(
        {k: v for k, v in list(evaluation.classifiers().items())[:3]}, X, y,
        n_splits=3)
    assert table['test accuracy'].is_monotonic_decreasing
