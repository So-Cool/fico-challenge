# Jupyter Notebooks

The FICO Explainable Machine Learning Challenge (2018) used anonymised credit
applications for a home equity line of credit (HELOC):
10,459 applicants described by 23 features from their credit bureau records and
annotated whether they repaid as agreed (`Good`) or not (`Bad`).
This series of notebooks cleans the data, builds a small, transparent decision
tree and explains its decisions with counterfactual statements:

1. [the **data** notebook][data] describes the features, their special
   values and simple baselines;
2. [the **cleaning** notebook][cleaning] describes special values, consistency
   checks and engineered features;
3. [the **the model** notebook][model] describes choosing the features and
   the decision tree; and
4. [the **explanations** notebook][explanations] describes why individual
   applicants were assessed as they were and how the model works as a whole.

[data]: ./01_data.ipynb
[cleaning]: ./02_cleaning.ipynb
[model]: ./03_model.ipynb
[explanations]: ./04_explanations.ipynb
