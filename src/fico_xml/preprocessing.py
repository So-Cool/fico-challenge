"""
Cleaning, consistency checks and feature engineering.

The same code processes the whole data set and individual applicants:

1. **Recoding.**
   The categories of the two delinquency features are recoded to a common
   scale: 0 for derogatory comment, 1 for 120+ days, 2 for 90 days,
   3 for 60 days, 4 for 30 days delinquent, 5 for unknown delinquency,
   6 for current and never delinquent.
2. **Special values.**
   Applicants with no credit record at all (every feature is -9) cannot be
   assessed.
   Otherwise, -9 (no record) and -8 (no usable value) become missing values;
   -7 (condition not met) becomes missing too, except for the months since
   the most recent credit check, whose value is guessed from the number of
   credit checks in the last 6 months (``MSinceMostRecentInqexcl7daysGuess``).
3. **Consistency checks.**
   Records that contradict themselves -- e.g., more accounts opened in the last
   12 months than in total -- are reported (:data:`RULES`).
4. **Feature engineering.**
   Percentages become fractions;
   percentages of the total number of accounts, Boolean indicators and
   one-hot encoded delinquency categories are added.

``legacy=True`` reproduces the original cleaning exactly;
the differences are listed in the changelog.
"""
import dataclasses

import numpy as np
import pandas as pd

from fico_xml.data import FEATURES, TARGET

__all__ = ['Rule', 'RULES', 'CleanData', 'clean', 'check', 'recode',
           'clean_special_values', 'engineer_features', 'NO_RECORD',
           'FEATURE_COLUMNS']

#: The reason given for applicants without any credit record.
NO_RECORD = 'no_record'
#: Features given as percentages (0-100) and turned into fractions.
PERCENTAGES = ['ExternalRiskEstimate', 'PercentTradesNeverDelq',
               'PercentInstallTrades', 'NetFractionRevolvingBurden',
               'NetFractionInstallBurden', 'PercentTradesWBalance']
#: Features whose -8 ("no usable value") becomes a missing value.
NO_USABLE_VALUE = ['MSinceOldestTradeOpen', 'NetFractionRevolvingBurden',
                   'NetFractionInstallBurden', 'NumRevolvingTradesWBalance',
                   'NumInstallTradesWBalance',
                   'NumBank2NatlTradesWHighUtilization',
                   'MSinceMostRecentDelq', 'PercentTradesWBalance']
INQUIRY_MONTHS = 'MSinceMostRecentInqexcl7days'
INQUIRY_GUESS = 'MSinceMostRecentInqexcl7daysGuess'
DELINQUENCY = ['MaxDelqEver', 'MaxDelq2PublicRecLast12M']
RATIOS = ['RevolvingTradesWBalance', 'InstallTradesWBalance',
          'SatisfactoryTrades', 'Trades60Ever2DerogPubRec',
          'Trades90Ever2DerogPubRec', 'TradesOpeninLast12M',
          'Bank2NatlTradesWHighUtilization']
BOOLEANS = ['BoolNverDeliquentCalc', 'BoolAlwaysSatisfactoryCalc',
            'BoolEverDeliquent90PlusCalc', 'BoolInq7DCalc',
            'BoolAllTradesPast12MCalc', 'BoolTradePast12MCalc']


def _feature_columns(legacy):
    base = [f for f in FEATURES if f not in DELINQUENCY]
    columns = base + [INQUIRY_GUESS]
    if legacy:
        columns.append('PercentTradesNeverDelqCalc')
    columns += ['Perc{}Calc'.format(r) for r in RATIOS]
    columns += ['NumInqLast7DCalc'] + BOOLEANS
    columns += ['{}_{}'.format(f, i) for f in DELINQUENCY for i in range(7)]
    return columns + DELINQUENCY


#: The columns of the cleaned data (:func:`clean`).
FEATURE_COLUMNS = _feature_columns(legacy=False)


@dataclasses.dataclass(frozen=True)
class Rule(object):
    """A consistency check: ``violated(data)`` marks inconsistent rows."""
    name: str
    message: str
    violated: object


#: The consistency checks (in the order of the original analysis).
RULES = [
    Rule('recent_account_older_than_oldest',
         'Your most recent credit account is older than your oldest one.',
         lambda d: (d['MSinceOldestTradeOpen']
                    < d['MSinceMostRecentTradeOpen'])),
    Rule('average_age_above_oldest',
         'Your credit accounts are on average older than your oldest one.',
         lambda d: d['MSinceOldestTradeOpen'] < d['AverageMInFile']),
    Rule('worse_delinquency_recently_than_ever',
         'Your worst payment delay in the last 12 months is worse than your '
         'worst payment delay ever.',
         lambda d: d['MaxDelqEver'] > d['MaxDelq2PublicRecLast12M']),
    Rule('fewer_60_than_90_days_overdue',
         'Fewer of your accounts were ever 60 or more days overdue than 90 or '
         'more days overdue.',
         lambda d: d['NumTrades60Ever2DerogPubRec']
         < d['NumTrades90Ever2DerogPubRec']),
    Rule('delinquency_inconsistent_with_60_days',
         'Your worst payment delay ever is shorter than 60 days, but some of '
         'your accounts were 60 or more days overdue.',
         lambda d: (d['MaxDelqEver'] > 3)
         & (d['NumTrades60Ever2DerogPubRec'] > 0)),
    Rule('delinquency_inconsistent_with_90_days',
         'Your worst payment delay ever is shorter than 90 days, but some of '
         'your accounts were 90 or more days overdue.',
         lambda d: (d['MaxDelqEver'] > 2)
         & (d['NumTrades90Ever2DerogPubRec'] > 0)),
    Rule('more_recent_accounts_than_total',
         'More credit accounts were opened in the last 12 months than you '
         'have in total.',
         lambda d: d['NumTradesOpeninLast12M'] > d['NumTotalTrades']),
    Rule('no_accounts',
         'You have no credit accounts.',
         lambda d: d['NumTotalTrades'] == 0),
    Rule('more_accounts_with_balance_than_total',
         'You have more revolving and instalment accounts with a balance '
         'than credit accounts in total.',
         lambda d: (d['NumRevolvingTradesWBalance']
                    + d['NumInstallTradesWBalance']) > d['NumTotalTrades']),
    Rule('more_satisfactory_accounts_than_total',
         'You have more credit accounts in good standing than credit accounts '
         'in total.',
         lambda d: d['NumSatisfactoryTrades'] > d['NumTotalTrades']),
]


@dataclasses.dataclass
class CleanData(object):
    """
    The result of :func:`clean`.

    Attributes
    ----------
    data : pandas.DataFrame
        The cleaned and engineered features of the consistent applicants.
    target : pandas.Series or None
        Their class (``Good`` / ``Bad``) when the input had a target column.
    removed : pandas.Series
        The applicants that were removed and why: :data:`NO_RECORD` or the
        name of the (first) violated rule.
    violations : pandas.DataFrame
        Every rule violated by every applicant with a credit record
        (``True`` -- violated).
    """
    data: pd.DataFrame
    target: object
    removed: pd.Series
    violations: pd.DataFrame


def recode(features):
    """Recodes the delinquency categories to a common scale."""
    x = features.copy()
    x['MaxDelq2PublicRecLast12M'] = x['MaxDelq2PublicRecLast12M'].replace(
        {8: -5, 9: -5, 6: 5, 7: 6})
    x['MaxDelqEver'] = x['MaxDelqEver'].replace({9: -5, 1: -4}).replace(
        {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6})
    return x


def _guess_inquiry_months(count, legacy):
    """Months since the most recent credit check guessed from the number of
    checks in the last 6 months (none -> missing, 1 -> 6, ..., 6 -> 1,
    more -> 0)."""
    guess = pd.Series(np.nan, index=count.index)
    for n in range(1, 7):
        guess[count == n] = 7 - n
    guess[count > 6] = 0
    return guess


def clean_special_values(features, legacy=False):
    """
    Handles the special values (-9, -8, -7) of applicants with a record.

    Parameters
    ----------
    features : pandas.DataFrame
        Recoded features (see :func:`recode`) of applicants with a record.
    legacy : bool
        Reproduce the original cleaning (see the changelog).
    """
    x = features.replace(-9, np.nan)
    x['MSinceMostRecentDelq'] = x['MSinceMostRecentDelq'].replace(-7, np.nan)

    count = x['NumInqLast6Mexcl7days']
    months = x[INQUIRY_MONTHS]
    guess = months.copy()
    no_inquiry = months == -7
    guessed = _guess_inquiry_months(count, legacy)
    if legacy:  # missing counts kept the special value -7
        known = no_inquiry & count.notnull()
        guess[known] = guessed[known]
    else:
        guess[no_inquiry] = guessed[no_inquiry]
    x[INQUIRY_MONTHS] = months.replace(-7, np.nan)
    x[INQUIRY_GUESS] = guess

    x.loc[x['MaxDelq2PublicRecLast12M'] == -5, 'MaxDelq2PublicRecLast12M'] = 6
    for column in NO_USABLE_VALUE:
        x[column] = x[column].replace(-8, np.nan)

    no_usable = x[INQUIRY_MONTHS] == -8
    if legacy:  # only 0 and 1 credit checks were handled
        zero, one = no_usable & (count == 0), no_usable & (count == 1)
        x.loc[zero | one, INQUIRY_MONTHS] = np.nan
        x.loc[zero, INQUIRY_GUESS] = np.nan
        x.loc[one, INQUIRY_GUESS] = 6
    else:
        x.loc[no_usable, INQUIRY_MONTHS] = np.nan
        x.loc[no_usable, INQUIRY_GUESS] = guessed[no_usable]
    return x


def check(data):
    """
    Evaluates every consistency rule on every applicant.

    Parameters
    ----------
    data : pandas.DataFrame
        Features after :func:`clean_special_values`.

    Returns
    -------
    pandas.DataFrame
        ``True`` where an applicant violates a rule (one column per rule).
    """
    return pd.DataFrame({rule.name: rule.violated(data).fillna(False)
                         .astype(bool) for rule in RULES}, index=data.index)


def _legacy_consistent(data):
    """The rows kept by the original sequence of checks (which put back
    the rows removed by the 'fewer_60_than_90_days_overdue' check)."""
    violated = {rule.name: rule.violated for rule in RULES}
    d = data
    for name in ('recent_account_older_than_oldest',
                 'average_age_above_oldest',
                 'worse_delinquency_recently_than_ever'):
        d = d[~violated[name](d).fillna(False)]
    d4 = d
    d45 = d4[~violated['fewer_60_than_90_days_overdue'](d4).fillna(False)]
    m5 = violated['delinquency_inconsistent_with_60_days'](d45).fillna(False)
    d = d4.drop(d45[m5].index)  # (sic) d4, not d45
    for name in ('delinquency_inconsistent_with_90_days',
                 'more_recent_accounts_than_total', 'no_accounts',
                 'more_accounts_with_balance_than_total',
                 'more_satisfactory_accounts_than_total'):
        d = d[~violated[name](d).fillna(False)]
    return d.index


def engineer_features(data, legacy=False):
    """
    Turns percentages into fractions and adds the engineered features.

    Returns
    -------
    pandas.DataFrame
        The columns of :data:`FEATURE_COLUMNS` (``legacy`` adds
        ``PercentTradesNeverDelqCalc``).
    """
    x = data.copy()
    for column in PERCENTAGES:
        x[column] = x[column] / 100
    total = x['NumTotalTrades']
    for ratio in RATIOS:
        x['Perc{}Calc'.format(ratio)] = x['Num' + ratio] / total
    if legacy:  # (sic) the percentage of satisfactory trades, 0-100
        x['PercentTradesNeverDelqCalc'] = np.round(
            100 * x['NumSatisfactoryTrades'] / total).astype('int64')
    x['NumInqLast7DCalc'] = x['NumInqLast6M'] - x['NumInqLast6Mexcl7days']
    indicators = {
        'BoolNverDeliquentCalc': x['PercentTradesNeverDelq'] >= 1,
        'BoolAlwaysSatisfactoryCalc': x['PercSatisfactoryTradesCalc'] >= 1,
        'BoolEverDeliquent90PlusCalc': x['MaxDelqEver'] < 3,
        'BoolInq7DCalc': x['NumInqLast7DCalc'] > 0,
        'BoolAllTradesPast12MCalc': x['PercTradesOpeninLast12MCalc'] >= 1,
        'BoolTradePast12MCalc': x['NumTradesOpeninLast12M'] > 0}
    for name, value in indicators.items():
        x[name] = value.astype('int64')
    for column in DELINQUENCY:
        for i in range(7):
            x['{}_{}'.format(column, i)] = (x[column] == i).astype('int64')
    return x[_feature_columns(legacy)]


def clean(raw, legacy=False):
    """
    Cleans applicants (the whole data set or a few of them).

    Parameters
    ----------
    raw : pandas.DataFrame
        Raw data -- the 23 features and, optionally, the target.
    legacy : bool
        Reproduce the original cleaning exactly (see the changelog).

    Returns
    -------
    CleanData
    """
    target = raw[TARGET] if TARGET in raw.columns else None
    x = recode(raw[FEATURES])
    no_record = (x == -9).all(axis=1)
    x = clean_special_values(x[~no_record], legacy=legacy)
    violations = check(x)

    removed = pd.Series(NO_RECORD, index=raw.index[no_record], dtype=object)
    if legacy:
        kept = _legacy_consistent(x)
    else:
        kept = violations.index[~violations.any(axis=1)]
    inconsistent = violations.drop(kept)
    if len(inconsistent):
        first = inconsistent.idxmax(axis=1)
        first[~inconsistent.any(axis=1)] = 'legacy'
        removed = pd.concat([removed, first]).sort_index()
    data = engineer_features(x.loc[kept], legacy=legacy)
    return CleanData(
        data=data,
        target=None if target is None else target.loc[data.index],
        removed=removed, violations=violations)
