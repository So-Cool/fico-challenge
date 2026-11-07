import numpy as np
import pandas as pd
import pytest

from conftest import consistent_applicant
from fico_xml import evaluation, explain as E, model as M
from fico_xml import preprocessing
from fico_xml.data import FEATURES


def test_model(model, raw):
    assert model.features == M.MODEL_FEATURES
    assert model.classifier.get_depth() <= 7
    importance = model.feature_importance()
    assert set(importance.index) == set(M.MODEL_FEATURES)
    assert importance.is_monotonic_decreasing
    X, y = M.model_data(preprocessing.clean(raw))
    assert set(model.predict(X)) <= {'Good', 'Bad'}
    assert len(M.fingerprint(model.classifier)) == 64


def test_explanations_change_the_prediction(model, raw):
    for i in raw.index[:40]:
        e = E.explain(model, raw.loc[i], warn=False)
        assert e.explained, e.messages
        frame = pd.DataFrame([e.values])
        assert model.predict(frame)[0] == e.prediction
        # the closest change reverses (keeps) the decision
        for cf in e.contrastive[:5]:
            new = pd.DataFrame([dict(e.values, **cf.values)])
            assert model.predict(new)[0] != e.prediction
            assert len(cf.values) == cf.distance
        for cf in e.equivalent[:5]:
            new = pd.DataFrame([dict(e.values, **cf.values)])
            assert model.predict(new)[0] == e.prediction
        assert E.OUTCOMES[e.prediction] in e.text()


def test_options(model, raw):
    for i in raw.index[:30]:
        everything = E.explain(model, raw.loc[i], warn=False)
        consistent = E.explain(model, raw.loc[i], consistent_only=True,
                               warn=False)
        assert all(cf.consistent for cf in consistent.contrastive)
        assert len(consistent.contrastive) == sum(
            cf.consistent for cf in everything.contrastive)
        actionable = E.explain(model, raw.loc[i], actionable_only=True,
                               warn=False)
        assert actionable.actionable_only
        for cf in actionable.contrastive:
            for feature, value in cf.values.items():
                assert M.ACTIONABLE[feature] == 'decrease'
                assert value < actionable.values[feature]
            new = pd.DataFrame([dict(actionable.values, **cf.values)])
            assert model.predict(new)[0] != actionable.prediction
        if not actionable.contrastive:
            assert 'No possible change' in actionable.text()


def test_monotonicity_warning(model, raw):
    import warnings
    flagged = 0
    for i in raw.index[:40]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            e = E.explain(model, raw.loc[i])
        contradicting = [cf for cf in e.contrastive
                         if cf.contradicts_monotonicity]
        flagged += bool(contradicting)
        assert bool(contradicting) == any(
            issubclass(w.category, E.MonotonicityWarning) for w in caught)
        for cf in contradicting:
            for feature in cf.contradicts_monotonicity:
                step = cf.values[feature] - e.values[feature]
                towards_good = e.prediction == 'Bad'
                # the risk would move the wrong way
                assert (M.MONOTONICITY[feature] * step > 0) == towards_good
    # in a monotone tree a change against the constraints is never needed:
    # undoing it keeps the reversed decision
    monotone = M.train(preprocessing.clean(raw), monotone=True)
    for i in raw.index[:40]:
        e = E.explain(monotone, raw.loc[i], warn=False)
        for cf in e.contrastive:
            if not cf.contradicts_monotonicity:
                continue
            undone = {f: v for f, v in cf.values.items()
                      if f not in cf.contradicts_monotonicity}
            new = pd.DataFrame([dict(e.values, **undone)])
            assert monotone.predict(new)[0] != e.prediction


def test_describe():
    assert E.describe({'A': {'label': 'number of credit checks', 'is': 3,
                             'wannabe': ['at most 0']}}) == \
        'there been no credit checks (instead of 3)'
    assert E.describe({'A': {'label': 'total number of credit accounts',
                             'is': 3, 'wannabe': ['at least 5']}}) == \
        'the total number of credit accounts been at least 5 (instead of 3)'


def test_inconsistent_counterfactual(model):
    row = consistent_applicant(np.random.default_rng(5))
    row.update(NumTrades60Ever2DerogPubRec=2, NumTrades90Ever2DerogPubRec=2,
               MaxDelqEver=2, MaxDelq2PublicRecLast12M=0)
    e = E.explain(model, row, warn=False)
    for cf in e.contrastive + e.equivalent:
        sixty = cf.values.get('NumTrades60Ever2DerogPubRec')
        ninety = cf.values.get('NumTrades90Ever2DerogPubRec', 2)
        if sixty is not None and sixty < ninety:
            assert not cf.consistent


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
    X, y = M.model_data(preprocessing.clean(raw))
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
