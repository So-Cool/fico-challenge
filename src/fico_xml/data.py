"""
The FICO HELOC data: download, verification and loading.

The data are not part of this package.
:func:`download` fetches them from a public copy of the official release
(pinned to a commit, so that later changes of that repository cannot affect
them);
copies from the official FICO release or from Kaggle can be placed in
the data directory instead.
All copies are verified with a checksum of their *content*
(independent of line endings).

The data directory is ``$FICO_XML_DATA`` if set, else ``~/.cache/fico-xml``.
"""
import functools
import hashlib
import json
import os
import urllib.request

import pandas as pd

__all__ = ['DATA_URL', 'DATA_FILE', 'CONTENT_SHA256', 'data_home',
           'data_path', 'download', 'load_raw', 'content_checksum',
           'feature_metadata', 'FEATURES', 'TARGET']

#: A public copy of the official data, pinned to a commit.
DATA_URL = ('https://raw.githubusercontent.com/5teffen/FICO-xML-Challenge/'
            '056ca9505910db184c10d20457551ab48925e317/'
            'xML%20Challenge%20Dataset%20and%20Data%20Dictionary/'
            'heloc_dataset_v1.csv')
DATA_FILE = 'heloc_dataset_v1.csv'
#: SHA-256 of the content of the official data (see :func:`content_checksum`).
CONTENT_SHA256 = \
    'f66d77f4e4e27c11205ba6258008f5e071f1f09724f4893e9ff2840a4e99f259'

_RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'resources')


@functools.lru_cache(maxsize=None)
def _metadata():
    with open(os.path.join(_RESOURCES, 'features.json')) as f:
        return json.load(f)


def feature_metadata():
    """
    The metadata of the features (loaded from ``resources/features.json``).

    Returns
    -------
    dict
        ``target``, ``special_values`` and ``features`` -- for every feature
        its official ``description``, a plain-English ``label``, its
        ``monotonicity`` constraint, ``kind`` and (ordinal features) the
        meaning of its ``categories``.
    """
    return json.loads(json.dumps(_metadata()))


#: The 23 features in the order of the data file.
FEATURES = list(_metadata()['features'])
#: The name of the target column.
TARGET = _metadata()['target']['name']


def data_home(path=None):
    """The data directory (created if needed)."""
    if path is None:
        path = os.environ.get('FICO_XML_DATA',
                              os.path.join('~', '.cache', 'fico-xml'))
    path = os.path.expanduser(path)
    os.makedirs(path, exist_ok=True)
    return path


def data_path(path=None):
    """The location of the data file in the data directory."""
    return os.path.join(data_home(path), DATA_FILE)


def content_checksum(data):
    """
    SHA-256 of the content of the data (a data frame), independent of the
    formatting of the file it was read from (e.g., line endings).
    """
    payload = data.to_csv(index=False, lineterminator='\n')
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def download(path=None, url=DATA_URL, force=False):
    """
    Downloads the data into the data directory (unless already there).

    Parameters
    ----------
    path : str, optional
        The data directory (see :func:`data_home`).
    url : str
        Where to download the data from.
    force : bool
        Download even if the file exists.

    Returns
    -------
    str
        The location of the data file.

    Raises
    ------
    ValueError
        The downloaded data differ from the official data.
    """
    target = data_path(path)
    if os.path.isfile(target) and not force:
        return target
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    temporary = target + '.part'
    with open(temporary, 'wb') as f:
        f.write(payload)
    try:
        _verify(pd.read_csv(temporary))
    except ValueError:
        os.remove(temporary)
        raise
    os.replace(temporary, target)
    return target


def _verify(data):
    checksum = content_checksum(data)
    if checksum != CONTENT_SHA256:
        raise ValueError('The data differ from the official FICO HELOC data '
                         '(content checksum {}).'.format(checksum))


def load_raw(path=None, verify=True):
    """
    Loads the raw data (10,459 applicants, the target and 23 features).

    Parameters
    ----------
    path : str, optional
        A data file or a data directory (default: :func:`data_home`).
        The file is *not* downloaded automatically -- see :func:`download`.
    verify : bool
        Check that the data are the official data.

    Returns
    -------
    pandas.DataFrame
    """
    if path is not None and os.path.isfile(path):
        file = path
    else:
        file = data_path(path)
    if not os.path.isfile(file):
        raise FileNotFoundError(
            'The data file {} does not exist; download it with '
            'fico_xml.data.download() (see the README).'.format(file))
    data = pd.read_csv(file)
    if verify:
        _verify(data)
    return data
