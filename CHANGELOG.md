# Changelog

## 0.0.2

### Changes

* **Accepted or declined.**
  Explanations describe the outcome as *accepted* (`Good`) or *declined*
  (`Bad`) -- e.g., "The application would be accepted had ..." -- instead of
  the names of the classes.
* **Wording.**
  "Had the number of ... been at most 0." --> "Had there been no ..."
* **Overdue accounts.**
  The labels of the numbers of accounts ever 60 (90) or more days overdue now
  mention that they include derogatory public records.
* **Counterfactuals as objects.**
  `Explanation.contrastive` and `Explanation.equivalent` hold
  `Counterfactual` objects:
  the described changes, the *closest* change (`values` captures the value of
  every changed feature nearest to the applicant's), whether it gives a
  consistent record (`consistent`) and which features it moves against their
  monotonicity constraint (`contradicts_monotonicity`).
* **Consistent counterfactuals.**
  When `explain(..., consistent_only=True)` is set, counterfactuals keep the
  features they do not change, which can give a record that contradicts itself
  (e.g., fewer accounts 60 days overdue than 90 days overdue), which affects
  about 42% of all the class-contrastive counterfactuals of the original tree;
  the option keeps only the counterfactuals whose closest change passes the
  consistency checks.
* **Monotonicity.**
  The original tree does not obey the monotonicity constraints of the data
  dictionary, with about 42% of its class-contrastive counterfactuals moving a
  feature against its constraint (e.g., *more* accounts opened in the last
  12 months would lead to acceptance).
  Such counterfactuals are flagged (noted in the text) and `explain` issues
  a `MonotonicityWarning` (`warn=False` silences it).
  `fico_xml.model.MONOTONICITY` holds the constraints (the number of credit
  checks in the last 7 days, an engineered feature, is treated like the number
  of credit checks).
* **Actionable counterfactuals.**
  When `explain(..., actionable_only=True)` is set, only class-contrastive
  counterfactuals that change what an applicant can act on are generated.
  Those include paying off balances and applying for or opening less credit
  (the numbers of accounts with a balance, credit cards with high utilisation,
  credit checks and accounts opened in the last 12 months, all listed in
  `fico_xml.model.ACTIONABLE` and decreasing);
  the history (overdue and total accounts, accounts in good standing)
  cannot change.
* **A monotone tree.**
  The `model.train(..., monotone=True)` option trains a monotone tree and
  the new notebook compares it with the original tree:
  This new tree respects the constraints (no explanation needs a change against
  them) at the cost of about two points of accuracy.
* **"All other" delinquency code.**
  The six applicants whose worst payment delay in the last 12 months is coded
  "all other" (code 9) are recoded as "current and never delinquent", as in
  the original code, which is justified by their records:
  all of them are recorded as never delinquent everywhere else
  (the worst payment delay ever, no months since a delinquency and 100% of
  accounts never delinquent).
* **Notebooks.**
  Notebook 3 used the old names of the model constants and failed;
  the volatility table of notebook 4 (eight leaves with the same value)
  became a summary per class.
* **Challenge URL.**
  The README links the challenge announcement instead of the challenge website,
  which no longer exists.

## 0.0.1

### Compatibility

* The cleaning, the model and the explanations of the original analysis were
  rewritten as the `fico_xml` package;
  the explanations are computed by [CtreeX][ctreex] (version 0.0.4) instead of
  a custom version of the explainer.
* `fico_xml.preprocessing.clean(raw, legacy=True)` reproduces the data the
  original analysis produced
  (`all_preprocessed_data_fix_1.pkl`: 8,199 applicants, 53 columns)
  value for value;
  a test compares their checksums.
  On the FICO data the current cleaning keeps the same applicants with the
  same values;
  the original table only has one more column
  ([see below](#non-critical-fixes-always-applied)).
* Training the model reproduces the original decision tree
  (nodes, split features and thresholds fingerprinted by a test);
* The explanations of CtreeX's `legacy` (original) and `fixed` implementations
  are identical for all 7,084 applicants the model can explain
  (both kinds of counterfactuals);
  the fixed implementation is used throughout.
* The data are not distributed;
  `fico_xml.data.download()` fetches a public copy of the official release and
  every copy is verified with a checksum of its content.
* Tests run on synthetic applicants;
  tests on the FICO data run when the data are available.

### Bug fixes and behaviour changes

#### Critical fixes (always applied)

* **Checking applicants.**
  The original pipeline checked one applicant at a time and stopped at the
  first violated rule (`check` returned the first message for the whole table);
  `clean` processes any number of applicants and reports every rule each of
  them violates (`CleanData.violations`).
* **Applicants with missing values.**
  Three of the model's features can be missing after cleaning
  (no usable value in the record);
  the original pipeline failed for such applicants.
  `explain` reports that the model cannot assess them, naming the features.
* **Cross-validation.**
  `StratifiedKFold(shuffle=False, random_state=42)` is rejected by current
  scikit-learn;
  the seed had no effect without shuffling, so it was removed, which does not
  change the folds.

#### Non-critical fixes (always applied)

* **The sequence of checks.**
  The original checks removed the rows that failed the "60 versus 90 days
  overdue" check and then continued from the table *before* that removal,
  effectively putting the rows back.
  No applicant of the FICO data fails that check so the cleaned data are
  unaffected.
* **No usable months since the most recent credit check (-8).**
  The original cleaning handled only applicants with 0 or 1 credit checks in
  the last 6 months and left the special value -8 in the data
  (and in the guessed months) for 2 or more checks;
  these are now missing values and the months are guessed like for -7.
  No consistent applicant of the FICO data is affected.
* **Guessed months without a count.**
  A -7 ("no credit check") with a missing number of credit checks left the
  special value -7 in the guessed months;
  it is now missing.
  No consistent applicant of the FICO data is affected.
* **`PercentTradesNeverDelqCalc`.**
  The original table had a column `PercentTradesNeverDelqCalc` computed as
  `round(100 * NumSatisfactoryTrades / NumTotalTrades)`, i.e., the percentage
  of *satisfactory* accounts, which is a copy of
  `PercSatisfactoryTradesCalc` on a 0-100 scale.
  It is no longer created (`legacy=True` still creates it).
* **Messages.**
  The messages of the consistency checks were corrected ("deliquent",
  "ballance", "Your on average longer in the file ...") and rewritten for
  applicants.
* **Describing ranges.**
  Ranges were described with the split thresholds and a wrong comparison
  ("less than 0.5" for "at most 0.5");
  the counts are now described with whole numbers
  ("at most 0", "between 20 and 32").
* **Feature descriptions.**
  The plain-English descriptions of the original code (some ending with
  question marks) were replaced by checked, short labels used in
  the explanations (`resources/features.json`).
  The number of bank or national credit-card accounts with *high utilisation*
  was described as having a utilisation "greater than 75%";
  since the data dictionary gives no threshold, it is now described as *high*
  utilisation.

### Original behaviour

* The months since the most recent credit check are guessed from the number
  of credit checks in the last 6 months based on a simple heuristic
  (1 check -> 6 months, ..., 6 checks -> 1 month, more -> 0).
* The six applicants whose worst payment delay in the last 12 months is
  coded "all other" (code 9) are recoded as "current and never delinquent".

## pre 0.0.1

* **Python and package versions.**
  Python 3 with scikit-learn 0.19, pandas 0.22, seaborn and pydotplus
  (Graphviz) for plotting the tree.

[ctreex]: https://github.com/So-Cool/ctreex
