# Baseline Predictive Pipeline -- ETAI

# Duarte Fernandes 20231619

## Overview
This project predicts recidivism, specifically, whether a defendant released from jail will commit another crime within two years, modeled directly on the COMPAS risk-assessment tool used by U.S. courts for bail, sentencing, and parole decisions. The algorithm takes details like prior offenses and age to assign an individual a risk score intended to help judges gauge public safety.

What makes this problem so compelling is the human impact of algorithmic justice, combined with a notorious real-world failure in algorithmic fairness. In 2016, an investigation revealed that while the system had similar overall accuracy across racial groups, its errors were severely skewed, as in Black defendants were nearly twice as likely to be falsely flagged as high-risk, while white defendants were far more likely to be mislabeled as low-risk before reoffending.

What drew me to this is how easy it is to think computers are neutral, when in reality they just inherit our own messes. If the real world has biased policing, the training data reflects that, and a basic model will happily turn those historical patterns into unfair decisions that ruin lives. As a student, digging into this project feels eye-opening because it forces me to deal with a harsh reality that you can't just throw raw data into a model and hope it's fair. Learning how to spot where that bias sneaks in and actually trying to fix it in code feels like using data science for something that genuinely matters.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check) and `src/preprocessing.py` (leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, **and** the train/test split itself, all in the one file rather than split across two) replace the old naive `dropna()` / `pd.get_dummies()` preprocessing; encoder/scaler pair (count encoding + robust scaling) chosen by an empirical grid over 15 repeated splits, checked against the runner-up with a paired comparison so the win isn't just noise; three redundant columns (found via correlation + VIF) dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections -- see "Preprocessing decisions" below. |
| 4 | Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly | A **locked final test set** (20%, stratified, seed 42) is set aside by `split_dev_test()` (replaces `split_train_test()`) and never scored; models are now judged by **stratified 5-fold cross-validation** of the whole pipeline (preprocessing + model) on the development set, reported per fold with mean ± std and the train-validation gap; the classification report and fairness check now use out-of-fold predictions; target encoding switched to scikit-learn's cross-fitting `TargetEncoder` (a row's own label never leaks into its own encoding), encoder/scaler set by hand in `config.yaml` (target encoding + robust scaling, reasons in the comments); **two fixes** in `clean_dataset()`: genuine `NaN`s in categorical columns were being turned into the string `"nan"` (a fake category), so 229 `c_charge_degree` gaps were never imputed or flagged -- fixed in `config.yaml` alone: `"nan"` added to `diagnostics.placeholder_tokens` (the category cleanup's last step turns listed tokens into `NaN`, after its text conversion); and it no longer drops rows -- de-duplication moved to a separate, training-only `drop_duplicate_rows()` (run before the dev/test split), so the same cleaning can run on new data where every row needs a prediction; `src/data_diagnostics.py` removed -- its one cleaning function (`flag_invalid_values`) moved into `preprocessing.py`, and the EDA-only checks (missingness test, duplicate counts) live in the EDA notebooks, not in every pipeline run; `dummy` (majority-class) model added as the floor to beat, and `random_forest` registered (sensible defaults, untuned); the final model is refit on the whole development set after CV; `config.yaml` gains `test_set` and `cv` sections -- see "Model evaluation" below |
| 5 | Hyperparameter tuning without leakage | New `src/tuning.py`: hyperparameters are tuned with **Optuna** (seeded TPE sampler) on the development set only -- every candidate is scored by stratified 5-fold CV of the **whole** pipeline (preprocessing re-fit in every fold of every trial), on the same folds for every candidate; the tuning procedure is estimated honestly by **nested cross-validation** (the tuning repeated inside each outer CV fold, scored on the outer fold it never saw), whose out-of-fold predictions now feed the classification report and fairness check; the reported "best trial" score is shown next to the nested one as an optimism check; the final model is the tuned pipeline refit on all development rows; `build_pipeline()` in `src/model.py` now builds the preprocessing + model `Pipeline` (one place to add a learned step later); `config.yaml` gains a `tuning` section (on/off, trials, inner folds, seed, one search space per model type) and the model becomes the decision tree; `optuna` added to `requirements.txt`. The locked test set is still untouched -- see "Hyperparameter tuning" below |
| 6 | Ensembles in depth: random forest, oblique random forest, AdaBoost | Two new model types in `src/model.py`: `oblique_random_forest` (a forest of oblique trees built from scikit-learn parts -- every tree splits on its own random sparse linear combinations of the features, via `SparseRandomProjection` inside `BaggingClassifier`) and `adaboost` (AdaBoost over decision trees of a chosen depth; stumps by default), both registered in `_MODELS`; the random forest is now the model `main.py` runs (200 trees, all CPU cores); `config.yaml` gains search spaces for the three ensembles -- the forests tune tree size (`max_depth`, `min_samples_leaf`) and decorrelation (`max_features` / number of combinations) but not the number of trees, AdaBoost tunes its base trees' depth, its number of rounds and its learning rate; `corrected_paired_test()` in `src/evaluate.py` compares two models scored on the same folds (Nadeau & Bengio's corrected paired t-test), so a "winner" is only called when the difference is bigger than fold-to-fold noise; `scipy` listed in `requirements.txt` (it already came with scikit-learn, so nothing new to install) -- see "Ensembles" below |
| 7 | Gradient boosting + evaluating with probabilities | Three new model types in `src/model.py`: `gradient_boosting` (scikit-learn's `GradientBoostingClassifier`: each shallow tree fits the gradient of the log-loss left by the trees before it), `hist_gradient_boosting` (the same idea on binned features: much faster on large data, and it handles missing values itself) and `naive_bayes` (`GaussianNB`, a fast probabilistic baseline); gradient boosting is now the model `main.py` runs, each with its search space in `config.yaml`; evaluation now reports **several metrics** (accuracy, balanced accuracy, precision, recall, F1, F-beta, ROC-AUC, PR-AUC) and one **selection metric** (`cv.selection_metric`, ROC-AUC) that tuning maximises and runs are ranked by; out-of-fold **probabilities** (not just labels) feed the reports, with a confusion matrix; `class_weight: balanced` in the commented examples of the models that support it; new `save_scores()` in `src/results.py`: every run adds a row to `results/runs.csv` (the model and its settings, the score on each fold, the no-model floor and the untuned score on the same folds), so any two runs on the same data and folds can be compared with `corrected_paired_test()`; the printed output was reorganised to be easier to read, with the full tables in the results file (optional: keep your own format if you prefer) -- see "Model evaluation" below |
|---|---|---|

### Metrics
`cv.metrics` lists what every run reports; `cv.selection_metric` is the one that matters: tuning maximises it and the scoreboard ranks runs by it. The default is **ROC-AUC**: it scores how well the model *ranks* people by risk, without choosing a threshold, so COMPAS's own 1-10 risk score can be judged the same way. Accuracy, precision, recall and F1 depend on the threshold that turns a probability into a yes/no (0.5 in this pipeline). `fbeta_<beta>` is F-beta: `fbeta_2` counts recall twice as much as precision.

**`f1`** was chosen because the downstream pipeline and fairness audit evaluate hard binary classifications locked at a fixed 0.5 decision threshold. Optimizing directly on $F_1$ forces hyperparameter tuning to maximize the exact precision–recall trade-off evaluated in the classification report, rather than favoring models with high global rank-ordering that may be miscalibrated around the 0.5 boundary. This keeps model selection aligned with the real-world bail decisions and racial False Positive Rate disparities tracked in the audit

### Scoreboard
Every `python main.py` adds one row to `results/runs.csv` (`save_scores()` in `src/results.py`): the run id, the data file, the model, whether it was tuned, the encoder and scaler, the model's params (and the tuned ones), the selection metric with its mean and std, the **untuned** score of the same model, the **no-model floor** (a dummy on the same folds) and the score on each fold (`fold_1` ... `fold_5`, from nested CV when tuning is on). The console prints the runs on the same data and metric, best first. The full report of each run is in `results/run_<run id>.txt`.

| run | model | tuned | f1 | std | untuned | floor |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 20261107_224121 | random_forest | True | 0.641 | 0.015 | 0.605 | 0.000 |
| 20261107_223731 | logistic_regression | True | 0.637 | 0.013 | 0.633 | 0.000 |
| 20261107_223639 | hist_gradient_boosting | True | 0.637 | 0.018 | 0.630 | 0.000 |
| 20261107_224335 | decision_tree | True | 0.625 | 0.016 | 0.569 | 0.000 |
| 20261107_221147 | adaboost | True | 0.619 | 0.019 | 0.599 | 0.000 |
| 20261107_224415 | oblique_random_forest | True | 0.616 | 0.011 | 0.589 | 0.000 |
| 20261107_223541 | gradient_boosting | True | 0.611 | 0.016 | 0.607 | 0.000 |
| 20261107_223940 | dummy | True | 0.448 | 0.020 | 0.000 | 0.000 |
| 20261107_223500 | naive_bayes | True | 0.403 | 0.012 | 0.403 | 0.000 |

### Model Evaluation

| Model | Holdout accuracy (W3) | CV accuracy (mean ± std) | CV train-val gap |
|---|---|---|---|
| Dummy | — | 0.549 ± 0.0 | ±0.0 |
| Logistic regression | 0.657 | 0.673 ± 0.014 | +0.002 |
| Decision tree | 0.613 | 0.594 ± 0.022 | +0.105 |
| Random forest | — | 0.644 ± 0.006 | +0.095 |
| Oblique random forest | — | 0.646 ± 0.015 | +0.118 |
| AdaBoost | — | 0.676 ± 0.011 | +0.007 |

The cross-validation validation accuracy (mean ± std) is the most trustworthy metric because it averages performance over five stratified splits, mitigating the variance and potential split bias of a single holdout set. Under default settings, AdaBoost (0.676) and Logistic Regression (0.673) demonstrate the best baseline performance and the tightest generalization gaps (+0.007 and +0.002). In contrast, unregularized Decision Trees and default Random Forests overfit the training splits considerably, showing gaps between +0.095 and +0.118.

### Hyperparameter tuning
How the pipeline tunes (Optuna, search space, number of trials, inner folds) and why it doesn't leak.

### Hyperparameter tuning
How the pipeline tunes (Optuna, search space, number of trials, inner folds) and why it doesn't leak.

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Decision tree | 0.594 ± 0.022 | 0.672 ± 0.011 | 0.674 | -0.000 | `criterion='entropy'`, `max_depth=10`, `min_samples_leaf=55` |
| Logistic regression | 0.673 ± 0.014 | — | — | — | — |
| Random forest | 0.644 ± 0.006 | 0.681 ± 0.018 | 0.685 | +0.000 | `max_depth=16`, `max_features='sqrt'`, `min_samples_leaf=24` |
| AdaBoost | 0.676 ± 0.011 | 0.682 ± 0.015 | 0.685 | +0.000 | `estimator__max_depth=4`, `n_estimators=55`, `learning_rate=0.167` |

Decision tree (+0.078) and random forest (+0.037) gained substantially from tuning, with improvements far exceeding their fold-to-fold standard deviations, while AdaBoost saw a modest increase of +0.006. Tuning changed the top-performing model: while logistic regression was the best un-tuned baseline (0.673), tuned AdaBoost (0.682) and tuned random forest (0.681) now take the lead. For the best model (AdaBoost), the value to report is the honest nested CV score of 0.682 ± 0.015 rather than the optimistic best trial score of 0.685, as nested CV prevents data leakage by keeping outer validation folds completely isolated from hyperparameter selection.

### Ensembles

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.673 ± 0.014 (+0.002) | — | — | — |
| Decision tree | 0.594 ± 0.022 (+0.105) | 0.672 ± 0.011 (+0.010) | 0.674 | `criterion='entropy'`, `max_depth=10`, `min_samples_leaf=55` |
| Random forest | 0.644 ± 0.006 (+0.095) | 0.681 ± 0.018 (+0.007) | 0.685 | `max_depth=16`, `max_features='sqrt'`, `min_samples_leaf=24` |
| Oblique random forest | 0.646 ± 0.015 (+0.118) | 0.680 ± 0.012 (+0.011) | 0.682 | `project__n_components=8`, `tree__max_depth=12`, `tree__min_samples_leaf=149` |
| AdaBoost | 0.676 ± 0.011 (+0.007) | 0.682 ± 0.015 (+0.007) | 0.685 | `estimator__max_depth=4`, `n_estimators=55`, `learning_rate=0.167` |

AdaBoost is the top model with an honest nested CV accuracy of 0.682 ± 0.015, followed closely by Random Forest at 0.681 ± 0.018 and Oblique Random Forest at 0.680 ± 0.012. However, its lead over both random forest variants (0.001 to 0.002) is well within the fold-to-fold standard deviations (±0.012 to ±0.018), making the ensemble performances practically equivalent within noise. The chosen hyperparameters reflect how each architecture operates: AdaBoost uses shallow base estimators (max_depth=4) paired with a conservative learning_rate=0.167 and fewer iterations (n_estimators=55) to boost sequentially without overfitting noisy labels, while Random Forest relies on deeper independent trees (max_depth=16) stabilized by large leaf minimums (min_samples_leaf=24) to average out variance. For the final deliverable, tuned AdaBoost should be reported at its honest nested CV accuracy of 0.682 ± 0.015 with a generalization gap of +0.007.

## Preprocessing decisions

| Column(s) | Issue found | Mechanism | What was done |
| :--- | :--- | :--- | :--- |
| `age` | 2.0% missing | MCAR | median impute, no indicator needed |
| `juv_fel_count` | 3.0% missing | MCAR | median impute, no indicator needed |
| `priors_count` | ~7% missing (incl. placeholder tokens) | MNAR -- tied to `age_cat` | median impute + `priors_count_was_missing` flag |
| `c_charge_degree` | 3.2% missing | MNAR -- tied to `age_cat` | mode impute + `c_charge_degree_was_missing` flag |
| `race` | ~1% missing (placeholder tokens) | MCAR | mode impute, no indicator (excluded from model features anyway) |
| `sex` | ~1.5% missing (incl. placeholder tokens) | MCAR | mode impute, no indicator needed |
| `age`, `decile_score`, `juv_fel_count`, `priors_count` | invalid values (out-of-range or negative) | domain rule | converted to `NaN` before imputation |
| `sex` / `race` / `c_charge_degree` / `score_text` | inconsistent category spelling (casing, whitespace, abbreviations) | data entry | canonicalized to one spelling per category |
| whole rows | 72 exact-duplicate rows, all sharing a repeated `id` | data entry | dropped, kept first occurrence |
| `prior_offenses`, `age_in_months`, `juvenile_total` | redundant with other columns (correlation r=1.00, or -- for `juvenile_total` -- an exact sum caught only by VIF) | multicollinearity | dropped |

**Encoder/scaler pair:** chosen empirically -- 4 encoders (one-hot, ordinal, count, target) × 4 scalers (none, standard, min-max, robust), scored by mean accuracy across 15 repeated train/test splits with logistic regression. **Target encoding + standard scaling won**, though a paired comparison against the runner-up (same 15 splits, per-split difference) showed the margin was within noise.

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
