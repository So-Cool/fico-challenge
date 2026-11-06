import numpy as np
import pandas as pd
import pytest

from conftest import consistent_applicant
from fico_xml import preprocessing as pp
from fico_xml.data import FEATURES


def one(**changes):
    row = consistent_applicant(np.random.default_rng(1))
    row.update(changes)
    return pd.DataFrame([row])[FEATURES]


def test_consistent_data_is_kept(raw):
    result = pp.clean(raw)
    assert len(result.removed) == 0 and len(result.data) == len(raw)
    assert list(result.data.columns) == pp.FEATURE_COLUMNS
    assert result.target.index.equals(result.data.index)
    assert not result.violations.any().any()


def test_no_record():
    result = pp.clean(pd.DataFrame([{f: -9 for f in FEATURES}]))
    assert list(result.removed) == [pp.NO_RECORD] and result.data.empty


@pytest.mark.parametrize('rule, changes', [
    ('recent_account_older_than_oldest',
     {'MSinceOldestTradeOpen': 10, 'MSinceMostRecentTradeOpen': 20,
      'AverageMInFile': 5}),
    ('average_age_above_oldest',
     {'MSinceOldestTradeOpen': 10, 'MSinceMostRecentTradeOpen': 1,
      'AverageMInFile': 30}),
    ('fewer_60_than_90_days_overdue',
     {'NumTrades60Ever2DerogPubRec': 1, 'NumTrades90Ever2DerogPubRec': 2,
      'MaxDelqEver': 2, 'MaxDelq2PublicRecLast12M': 0}),
    ('more_recent_accounts_than_total',
     {'NumTotalTrades': 3, 'NumTradesOpeninLast12M': 5,
      'NumSatisfactoryTrades': 1, 'NumRevolvingTradesWBalance': 0,
      'NumInstallTradesWBalance': 0}),
    ('no_accounts', {'NumTotalTrades': 0, 'NumTradesOpeninLast12M': 0,
                     'NumSatisfactoryTrades': 0,
                     'NumRevolvingTradesWBalance': 0,
                     'NumInstallTradesWBalance': 0}),
    ('more_satisfactory_accounts_than_total',
     {'NumTotalTrades': 3, 'NumSatisfactoryTrades': 5,
      'NumTradesOpeninLast12M': 0, 'NumRevolvingTradesWBalance': 0,
      'NumInstallTradesWBalance': 0}),
])
def test_rules(rule, changes):
    result = pp.clean(one(**changes))
    assert result.data.empty
    assert result.violations.iloc[0][rule]


def test_delinquency_rules():
    # raw MaxDelqEver 8 = never delinquent, but 60+ days overdue accounts
    violated = pp.clean(one(MaxDelqEver=8, MaxDelq2PublicRecLast12M=7,
                            NumTrades60Ever2DerogPubRec=1,
                            NumTrades90Ever2DerogPubRec=1)).violations.iloc[0]
    assert violated['delinquency_inconsistent_with_60_days']
    assert violated['delinquency_inconsistent_with_90_days']
    # worse in the last 12 months (raw 0 = derogatory) than ever (raw 8)
    assert pp.clean(one(MaxDelqEver=8, MaxDelq2PublicRecLast12M=0,
                        NumTrades60Ever2DerogPubRec=0,
                        NumTrades90Ever2DerogPubRec=0)).violations.iloc[0][
        'worse_delinquency_recently_than_ever']


def test_special_values():
    x = pp.clean(one(NumRevolvingTradesWBalance=-8, MSinceMostRecentDelq=-7,
                     MSinceMostRecentInqexcl7days=-7, NumInqLast6M=3,
                     NumInqLast6Mexcl7days=2)).data.iloc[0]
    assert np.isnan(x['NumRevolvingTradesWBalance'])
    assert np.isnan(x['MSinceMostRecentDelq'])
    assert np.isnan(x['MSinceMostRecentInqexcl7days'])
    assert x['MSinceMostRecentInqexcl7daysGuess'] == 5  # 2 checks -> 5


@pytest.mark.parametrize('count, legacy, expected', [
    (3, False, 4), (3, True, -8), (1, True, 6), (0, False, np.nan)])
def test_no_usable_inquiry_months(count, legacy, expected):
    x = pp.clean(one(MSinceMostRecentInqexcl7days=-8, NumInqLast6M=count,
                     NumInqLast6Mexcl7days=count), legacy=legacy).data.iloc[0]
    months = x['MSinceMostRecentInqexcl7days']
    if legacy and count > 1:  # (sic) original: left as the special value
        assert months == -8
    else:
        assert np.isnan(months)
    guess = x['MSinceMostRecentInqexcl7daysGuess']
    assert (np.isnan(guess) and np.isnan(expected)) or guess == expected


def test_engineered_features():
    x = pp.clean(one(NumTotalTrades=10, NumSatisfactoryTrades=10,
                     NumTradesOpeninLast12M=0, NumRevolvingTradesWBalance=2,
                     NumInstallTradesWBalance=1, NumInqLast6M=3,
                     NumInqLast6Mexcl7days=1, ExternalRiskEstimate=70,
                     MaxDelqEver=8, MaxDelq2PublicRecLast12M=7,
                     NumTrades60Ever2DerogPubRec=0,
                     NumTrades90Ever2DerogPubRec=0)).data.iloc[0]
    assert x['ExternalRiskEstimate'] == 0.7
    assert x['PercRevolvingTradesWBalanceCalc'] == 0.2
    assert x['NumInqLast7DCalc'] == 2 and x['BoolInq7DCalc'] == 1
    assert x['BoolAlwaysSatisfactoryCalc'] == 1
    assert x['MaxDelqEver'] == 6 and x['MaxDelqEver_6'] == 1
    assert x[['MaxDelqEver_{}'.format(i) for i in range(6)]].sum() == 0


def test_legacy_columns(raw):
    legacy = pp.clean(raw, legacy=True).data
    assert 'PercentTradesNeverDelqCalc' in legacy.columns
    assert 'PercentTradesNeverDelqCalc' not in pp.clean(raw).data.columns
