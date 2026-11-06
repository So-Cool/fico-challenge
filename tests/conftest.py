"""Synthetic applicants (no real data are distributed with the package)."""
import numpy as np
import pandas as pd
import pytest

from fico_xml.data import FEATURES, TARGET


def consistent_applicant(rng):
    """A random raw record that passes every consistency check."""
    total = int(rng.integers(1, 40))
    satisfactory = int(rng.integers(0, total + 1))
    sixty = int(rng.integers(0, 3))
    ninety = int(rng.integers(0, sixty + 1))
    worst_ever = int(rng.integers(0, 3)) if ninety else (
        int(rng.integers(0, 4)) if sixty else int(rng.integers(0, 7)))
    revolving = int(rng.integers(0, total + 1))
    install = int(rng.integers(0, total - revolving + 1))
    inquiries = int(rng.integers(0, 6))
    excluding = int(rng.integers(0, inquiries + 1))
    oldest = int(rng.integers(12, 400))
    return {
        'ExternalRiskEstimate': int(rng.integers(40, 95)),
        'MSinceOldestTradeOpen': oldest,
        'MSinceMostRecentTradeOpen': int(rng.integers(0, oldest + 1)),
        'AverageMInFile': int(rng.integers(1, oldest + 1)),
        'NumSatisfactoryTrades': satisfactory,
        'NumTrades60Ever2DerogPubRec': sixty,
        'NumTrades90Ever2DerogPubRec': ninety,
        'PercentTradesNeverDelq': int(rng.integers(50, 101)),
        'MSinceMostRecentDelq': int(rng.integers(0, 80)),
        # raw codes: MaxDelqEver 2-8, MaxDelq2PublicRecLast12M 0-7
        'MaxDelq2PublicRecLast12M': [0, 1, 2, 3, 4, 5, 7][
            int(rng.integers(worst_ever, 7))],
        'MaxDelqEver': worst_ever + 2,
        'NumTotalTrades': total,
        'NumTradesOpeninLast12M': int(rng.integers(0, total + 1)),
        'PercentInstallTrades': int(rng.integers(0, 101)),
        'MSinceMostRecentInqexcl7days': int(rng.integers(0, 24)),
        'NumInqLast6M': inquiries,
        'NumInqLast6Mexcl7days': excluding,
        'NetFractionRevolvingBurden': int(rng.integers(0, 101)),
        'NetFractionInstallBurden': int(rng.integers(0, 101)),
        'NumRevolvingTradesWBalance': revolving,
        'NumInstallTradesWBalance': install,
        'NumBank2NatlTradesWHighUtilization': int(rng.integers(0, 4)),
        'PercentTradesWBalance': int(rng.integers(0, 101)),
    }


def synthetic_data(n=600, seed=0):
    """Consistent applicants whose class depends on a few features."""
    rng = np.random.default_rng(seed)
    rows = [consistent_applicant(rng) for _ in range(n)]
    data = pd.DataFrame(rows)[FEATURES]
    good = ((data['NumTrades60Ever2DerogPubRec'] == 0)
            & (data['NumBank2NatlTradesWHighUtilization'] < 2)) \
        | (data['NumSatisfactoryTrades'] > 25)
    noise = rng.random(n) < 0.05
    data.insert(0, TARGET, np.where(good ^ noise, 'Good', 'Bad'))
    return data


@pytest.fixture(scope='session')
def raw():
    return synthetic_data()


@pytest.fixture(scope='session')
def model(raw):
    from fico_xml import model as M, preprocessing as pp
    return M.train(pp.clean(raw))
