"""AI-assisted implementation: OpenAI Codex / GPT-6, 2026-10-24.
Nested preprocessing/tuning and paired out-of-fold prediction. No raw-score claim.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from .common import ecdf_grades, preprocessing, progress, save_csv, write_json


def model_specs(cfg):
    def pipe(model):
        return Pipeline([("prep", preprocessing()), ("model", model)])
    return {
        "mean": (pipe(DummyRegressor(strategy="mean")), {}),
        "ridge": (pipe(Ridge()), {"model__alpha": cfg["ridge_alphas"]}),
        "pls": (pipe(PLSRegression(scale=False)), {"model__n_components": cfg["pls_components"]}),
        "extra_trees": (pipe(ExtraTreesRegressor(n_estimators=cfg["extra_trees"], min_samples_leaf=3, max_features=.8, random_state=cfg["seed"], n_jobs=2)), {}),
    }


def fitted_model(name, X, y, cfg):
    estimator, params = model_specs(cfg)[name]
    if params:
        estimator = GridSearchCV(estimator, params, scoring="neg_mean_absolute_error", cv=KFold(cfg["inner_folds"], shuffle=True, random_state=cfg["seed"]), n_jobs=1, refit=True)
    estimator.fit(X, y)
    return estimator.best_estimator_ if params else estimator, estimator.best_params_ if params else {}


def metric_rows(y, pred, targets, model):
    return [{"model": model, "target": c,
             "mae": float(mean_absolute_error(y[:, j], pred[:, j])),
             "rmse": float(np.sqrt(mean_squared_error(y[:, j], pred[:, j]))),
             "r2": float(r2_score(y[:, j], pred[:, j]))}
            for j, c in enumerate(targets)]


def run_q1(data, out, cfg):
    dest = Path(out) / "q1"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "private").mkdir(exist_ok=True)
    X, Y = data.X.to_numpy(), data.Y.to_numpy()
    targets = data.Y.columns.to_list()
    split = RepeatedStratifiedKFold(n_splits=cfg["outer_folds"], n_repeats=cfg["outer_repeats"], random_state=cfg["seed"])
    sums = {m: np.zeros_like(Y) for m in model_specs(cfg)}
    count = np.zeros(len(X), dtype=int)
    assignments, selected, fold_metrics, robustness = [], [], [], []
    grades = np.zeros_like(Y)
    for fold, (train, test) in enumerate(split.split(X, data.meta.tasting_category_fine)):
        progress(f"Q1: outer fold {fold + 1}/{cfg['outer_folds'] * cfg['outer_repeats']}")
        count[test] += 1
        assignments.extend({"repeat": fold // cfg["outer_folds"], "fold": fold % cfg["outer_folds"], "beer": int(data.meta.iloc[i].beer), "role": "validation"} for i in test)
        for name in sums:
            est, params = fitted_model(name, X[train], Y[train], cfg)
            pred = est.predict(X[test])
            sums[name][test] += pred
            selected.append({"outer_fold": fold, "model": name, "parameters": str(params)})
            fold_metrics.append({"outer_fold": fold, "model": name, "mae": float(mean_absolute_error(Y[test], pred)), "macro_r2": float(r2_score(Y[test], pred)), "n_validation": len(test)})
            if name == "ridge":
                grades[test] += ecdf_grades(Y[train], pred)
                for rate in cfg["missing_rates"]:
                    corrupted = X[test].copy()
                    rng = np.random.default_rng(cfg["seed"] + fold)
                    artificial_missing = rng.random(corrupted.shape) < rate
                    corrupted[artificial_missing] = np.nan
                    pp = est.predict(corrupted)
                    robustness.append({"outer_fold": fold, "additional_missing_fraction": rate, "mae": float(mean_absolute_error(Y[test], pp)), "mask_seed": cfg["seed"] + fold})
    if not np.all(count == cfg["outer_repeats"]):
        raise AssertionError("Every sample must have exactly one held-out prediction per repeat")
    predictions = {name: value / count[:, None] for name, value in sums.items()}
    rows = []
    for name, pred in predictions.items():
        rows += metric_rows(Y, pred, targets, name)
        frame = pd.concat([data.meta[["beer", "tasting_category_fine"]], pd.DataFrame(pred, columns=targets)], axis=1)
        save_csv(frame, dest / f"oof_{name}_continuous.csv")
    metrics = pd.DataFrame(rows)
    save_csv(metrics, dest / "metrics_by_target.csv")
    save_csv(metrics.groupby("model")[["mae", "rmse", "r2"]].mean().reset_index(), dest / "metrics_summary.csv")
    save_csv(pd.DataFrame(fold_metrics), dest / "metrics_by_fold.csv")
    save_csv(pd.DataFrame(assignments), dest / "split_assignments.csv")
    save_csv(pd.DataFrame(selected), dest / "inner_selected_parameters.csv")
    save_csv(pd.DataFrame(robustness), dest / "missingness_robustness.csv")
    styles = []
    for style in sorted(data.meta.tasting_category_fine.unique()):
        ix = np.flatnonzero(data.meta.tasting_category_fine.to_numpy() == style)
        styles.append({"style": style, "n": len(ix), "ridge_mae": float(mean_absolute_error(Y[ix], predictions["ridge"][ix]))})
    save_csv(pd.DataFrame(styles), dest / "metrics_by_style.csv")
    # Predeclared ridge is used downstream; do not choose the winner using OOF scores.
    model, params = fitted_model("ridge", X, Y, cfg)
    joblib.dump({"model": model, "features": data.X.columns.tolist(), "targets": targets}, dest / "private/model.joblib")
    save_csv(pd.DataFrame(model.named_steps["model"].coef_, index=targets, columns=data.X.columns).reset_index(names="target"), dest / "final_ridge_coefficients.csv")
    if cfg["export_assumption_grades"]:
        gg = np.rint(grades / count[:, None]).astype(int)
        frame = pd.concat([data.meta[["beer"]], pd.DataFrame(gg, columns=targets)], axis=1)
        frame["status"] = "ASSUMPTION_ONLY_training_ECDF_bins_not_absolute_sensory_scores"
        save_csv(frame, dest / "ASSUMPTION_percentile_grades_0_6.csv")
    write_json(dest / "status.json", {
        "continuous_prediction": "computed_repeated_out_of_fold_exploratory_baselines",
        "absolute_0_6_prediction": "UNRESOLVED_missing_raw_score_calibration",
        "proxy_grades": "empirical equal-frequency bins, not a recovery of the original 0-6 intensity",
        "primary_model": "ridge selected before evaluation; inner tuning only",
        "final_inner_parameters": params,
        "external_generalization": "not established; supplied targets already preprocessed by source authors",
        "intervals": "not calibrated; no nominal coverage claim is made",
    })
