"""The data module (synthetic data) and checks on the real data (skipped
unless the data are available)."""
import os

import pytest

from conftest import synthetic_data
from fico_xml import data
from fico_xml import model as M
from fico_xml import preprocessing as pp

#: content checksum of the original data
#: (all_preprocessed_data_fix_1.pkl: 8,199 applicants, 53 columns)
PREPROCESSED_ORIGINAL = \
    'c97053438ff333cb36a4dee550d948c42d80e7a84c3857fe9ef7746acb880b9c'


def test_metadata():
    meta = data.feature_metadata()
    assert len(data.FEATURES) == 23 and data.TARGET == 'RiskPerformance'
    for f in data.FEATURES:
        assert {'description', 'label', 'monotonicity', 'kind'} <= set(
            meta['features'][f])


def test_download_verifies(tmp_path):
    source = tmp_path / 'other.csv'
    synthetic_data(20).to_csv(source, index=False)
    with pytest.raises(ValueError):
        data.download(path=str(tmp_path / 'home'), url=source.as_uri())
    assert not os.path.exists(data.data_path(str(tmp_path / 'home')))
    with pytest.raises(FileNotFoundError):
        data.load_raw(str(tmp_path / 'empty'))
    # unverified loading of another file
    assert len(data.load_raw(str(source), verify=False)) == 20


AVAILABLE = os.path.isfile(data.data_path()) or \
    os.environ.get('FICO_XML_TEST_DOWNLOAD') == '1'
real = pytest.mark.skipif(not AVAILABLE, reason='the FICO data are not '
                          'available (see the README)')


@pytest.fixture(scope='module')
def real_raw():
    data.download()
    return data.load_raw()


@real
def test_reproduces_original_preprocessing(real_raw):
    import hashlib
    legacy = pp.clean(real_raw, legacy=True)
    table = legacy.data.astype(float).round(9).to_csv(lineterminator='\n')
    assert hashlib.sha256(table.encode()).hexdigest() == PREPROCESSED_ORIGINAL
    assert len(legacy.removed) == 10459 - 8199


@real
def test_reproduces_original_model(real_raw):
    model = M.train(pp.clean(real_raw))
    assert M.fingerprint(model.classifier) == M.ORIGINAL_FINGERPRINT


@real
def test_example_applicants(real_raw):
    from fico_xml.explain import explain
    model = M.train(pp.clean(real_raw))
    first = explain(model, real_raw.loc[34], warn=False)
    assert first.prediction == 'Bad'
    assert first.contrastive[0].distance == 1
    assert explain(model, real_raw.loc[678], warn=False).status == 'inconsistent record'
    assert explain(model, real_raw.loc[7890], warn=False).prediction == 'Good'
