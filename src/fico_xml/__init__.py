"""
fico-xml: explaining credit decisions on the FICO HELOC data.

>>> from fico_xml import data, preprocessing, model
>>> from fico_xml.explain import explain
>>> data.download()
>>> raw = data.load_raw()
>>> tree = model.train(preprocessing.clean(raw))
>>> print(explain(tree, raw.loc[34]).text())
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('fico-xml')
except PackageNotFoundError:  # pragma: no cover (not installed)
    __version__ = 'unknown'
