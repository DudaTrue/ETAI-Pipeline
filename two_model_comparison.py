import pandas as pd
from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/runs.csv")
folds = [c for c in runs.columns if c.startswith("fold_")]

# Filter only the XGBoost runs
xgb_runs = runs[runs["model"] == "xgboost"].reset_index(drop=True)

res = corrected_paired_test(
    xgb_runs.loc[0, folds], xgb_runs.loc[1, folds], test_train_ratio=1 / 4
)
print(res)