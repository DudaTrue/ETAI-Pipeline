"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> preprocess -> split -> train
    -> evaluate (train & test) -> save results
"""
import yaml
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import clean_dataset, split_features_target, build_preprocessor, split_train_test
from src.model import build_model
from src.evaluate import (
    cross_validate_pipeline, cv_report, oof_classification_report, fairness_report,
)
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = load_data(config["data"]["path"])
    df_clean = clean_dataset(df, config["diagnostics"])

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    X_train, X_test, y_train, y_test, extras_train, extras_test = split_train_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    pipeline = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])

    eval_config = config["cv"]
    eval_type = eval_config.get("type", "kfold")
    shuffle = eval_config.get("shuffle", True)

    if eval_type == "holdout":
        cv = StratifiedShuffleSplit(
            n_splits=1,
            test_size=eval_config.get("val_size", 0.2),
            random_state=eval_config.get("random_state", 42),
        )
    else:
        cv = StratifiedKFold(
            n_splits=eval_config["n_splits"], 
            shuffle=shuffle,
            random_state=eval_config.get("random_state") if shuffle else None
        )
    
    scoring = eval_config.get("scoring", "accuracy")

    # fold scores + out-of-fold predictions (each row predicted by the fold model that did NOT train on it)
    fold_scores, y_oof = cross_validate_pipeline(
        pipeline, X_train, y_train, cv, scoring, n_jobs=eval_config.get("n_jobs", 1)
    )

    report = cv_report(fold_scores, scoring)
    report += "\n\n" + oof_classification_report(y_train, y_oof)
    report += "\n" + fairness_report(
        y_train, y_oof, extras_train, sensitive_attr=config["data"]["sensitive_attr"]
    )

    pipeline.fit(X_train, y_train)

    refit = f"Final model: {config['model']['type']} refit on all {len(X_train)} development rows."
    print(refit)
    report += "\n" + refit + "\n"

    locked = (f"Locked test set: {len(X_test)} rows set aside, not evaluated. "
              f"Development set: {len(X_train)} rows.")
    print(locked)
    report += "\n" + locked + "\n"

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
