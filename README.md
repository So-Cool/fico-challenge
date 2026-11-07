# FICO XML Challenge

Explaining credit decisions on the FICO HELOC data for the 2018
[FICO Explainable Machine Learning Challenge][announcement].

A small, transparent decision tree assesses applications for a home equity
line of credit;
every decision is explained with counterfactual statements computed by
[CtreeX][ctreex]:

> The application is declined.
> The application would be accepted had there been no credit accounts that
> were ever 60 or more days overdue or have a derogatory public record
> (instead of 2).

Records without a credit history, records that contradict themselves and
records missing a value that the model needs are not assessed;
the applicant is told why.

## ToC

* [Data](#data)
* [Installation](#installation)
* [Quick start](#quick-start)
* [Notebooks](#notebooks)
* [Python package](#python-package)
* [Development](#development)

## Data

The data are **not** part of this repository.
They consist of 10,459 anonymised applications described by 23 features from
credit bureau records and whether the loan was repaid as agreed
(`Good` / `Bad`).
The data can be sourced from:

* the official FICO community explainable machine learning challenge
  (see the [announcement][announcement], registration required);
* [Kaggle][kaggle] (the `heloc_dataset_v1 (1).csv` file, account required); or
* the [5teffen/FICO-xML-Challenge][github-copy] GitHub repository
  (commit `056ca95`, default source) with `fico_xml.data.download()`;

The data are stored in `~/.cache/fico-xml` (or the directory in the
`FICO_XML_DATA` environment variable);
to use a copy from FICO or Kaggle, save it there as `heloc_dataset_v1.csv`.
Every copy is verified with a checksum of its content.

Nothing else needs to be downloaded or generated:
the cleaned data and the model are computed from the data file when needed.

## Installation

With [uv][uv]:

```sh
git clone https://github.com/So-Cool/fico-challenge.git
cd fico-challenge
uv sync                       # Python and the dependencies (locked)
uv run python -c "from fico_xml import data; data.download()"
```

The `graphviz` extra renders the tree with Graphviz (requires Graphviz binary);
plots use matplotlib otherwise.

## Quick start

```python
from fico_xml import data, model, preprocessing
from fico_xml.explain import explain

raw = data.load_raw()                         # after data.download()
tree = model.train(preprocessing.clean(raw))  # the decision tree model

explanation = explain(tree, raw.loc[34])      # one applicant's raw record
print(explanation.text())
explanation.contrastive                       # changes that flip the decision
explanation.equivalent                        # changes that keep it

# only consistent records; only changes the applicant can act on
explain(tree, raw.loc[2], consistent_only=True, actionable_only=True)

# a monotone tree
model.train(preprocessing.clean(raw), monotone=True)
```

## Notebooks

The Jupyter Notebooks -- placed in the [`notebooks` directory](./notebooks/) --
tell the whole story:

| Notebook | Content |
|----------|---------|
| `01_data`          | features, special values, baselines (external risk estimate alone reaches about 71% accuracy), delinquency and outcome |
| `02_cleaning`      | special values, consistency checks (1,672 records contradict themselves), feature engineering |
| `03_model`         | comparing models, ten count features, decision tree, feature importance |
| `04_explanations`  | explaining individual applicants, monotonicity, consistent and actionable counterfactuals, applicants that cannot be assessed, tree structure |
| `05_monotone_tree` | a tree that respects the monotonicity constraints, compared with the original one |

To run them:

```sh
uv sync --group notebooks
uv run jupyter lab notebooks/
```

## Python package

| Module | Purpose |
|--------|---------|
| `fico_xml.data`          | downloading, verifying and loading data; feature metadata (`resources/features.json` holds the official descriptions and monotonicity constraints complemented by plain-English labels) |
| `fico_xml.preprocessing` | `clean` handles recoding, special values, consistency checks (`RULES`) and feature engineering, for the whole data set or individual applicants |
| `fico_xml.model`         | `train` builds the decision tree and its CtreeX explainer |
| `fico_xml.evaluation`    | implements baselines and compares models (cross-validation) |
| `fico_xml.explain`       | `explain` explains the decision about an applicant with class-contrastive and class-equivalent counterfactuals; `plot_tree` and `graphviz_tree` are also available |

## Development

To run tests:

```sh
uv run pytest
```

The tests use synthetic applicants;
the tests on the FICO data
(that check data cleaning, tree fitting and example applicant explanations)
run when the data have already been downloaded or
when `FICO_XML_TEST_DOWNLOAD=1` is set.

The [changelog](CHANGELOG.md) lists the code fixes and validation.

[announcement]: https://community.fico.com/s/blog-post/a5Q8000000082UQEAY/fico1158
[ctreex]: https://github.com/So-Cool/ctreex
[github-copy]: https://github.com/5teffen/FICO-xML-Challenge/tree/056ca9505910db184c10d20457551ab48925e317/xML%20Challenge%20Dataset%20and%20Data%20Dictionary
[kaggle]: https://www.kaggle.com/datasets/averkiyoliabev/home-equity-line-of-creditheloc
[uv]: https://docs.astral.sh/uv/
