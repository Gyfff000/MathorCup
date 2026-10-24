"""AI-assisted extension: OpenAI Codex / GPT-6, 2026-10-24.
Scientific limitations and all failed experiments must remain visible.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp, norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from .audit import run_audit
from .common import ecdf_grades, preprocessing, progress, save_csv, sha256, write_json
from .data import CHEMICAL_DEFECTS
from .q1 import fitted_model, metric_rows
from .q3 import ConditionalPCAGaussian, biased_mmd, chemical_nonincrease


def conformal_quantile(errors, alpha):
    n = len(errors)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        raise ValueError("Calibration set too small for requested coverage")
    return np.sort(np.asarray(errors), axis=0)[rank - 1]


def assumption_grades(reference, pred):
    reference, pred = np.asarray(reference), np.asarray(pred)
    empirical = ecdf_grades(reference, pred)
    mu, sd = reference.mean(0), np.maximum(reference.std(0), 1e-8)
    gaussian = np.minimum(6, np.floor(7 * norm.cdf((pred - mu) / sd))).astype(int)
    low, high = np.quantile(reference, [.05, .95], axis=0)
    linear = np.rint(6 * np.clip((pred - low) / np.maximum(high - low, 1e-8), 0, 1)).astype(int)
    return np.stack([empirical, gaussian, linear])


def local_convex_samples(real, n, rng):
    """Conditional empirical local mixing; not a physical composition law."""
    real = np.asarray(real)
    scale = np.maximum(np.std(real, axis=0), 1e-8)
    dist = cdist(real / scale, real / scale)
    np.fill_diagonal(dist, np.inf)
    neighbors = np.argsort(dist, axis=1)[:, :min(3, len(real) - 1)]
    out = []
    for _ in range(n):
        anchor = rng.integers(len(real))
        partner = rng.choice(neighbors[anchor]) if len(real) > 1 else anchor
        weight = rng.uniform(.1, .9)
        out.append(weight * real[anchor] + (1 - weight) * real[partner])
    return np.asarray(out)


def run_extension(data, base, out, cfg, ext):
    manifest = pd.read_csv(base / "audit/source_manifest.csv")
    for row in manifest.itertuples():
        if sha256(data.root / row.file) != row.sha256:
            raise ValueError("Source version differs from stage one")
    run_audit(data, out, cfg)
    X, Y = data.X.to_numpy(), data.Y.to_numpy()
    labels = data.meta.tasting_category_fine.to_numpy()
    targets, features = data.Y.columns.tolist(), data.X.columns.tolist()
    for sub in ["q1", "q2", "q3"]:
        (out / sub).mkdir(exist_ok=True)
    splits = list(RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=cfg["seed"]).split(X, labels))
    sums = np.zeros_like(Y)
    counts = np.zeros(len(Y))
    permutation_sums = np.zeros((X.shape[1], Y.shape[1]))
    positive_folds = np.zeros_like(permutation_sums)
    grade_min, grade_max = np.full_like(Y, 6), np.zeros_like(Y)
    grade_disagreement, assignments = [], []
    for fold, (train, test) in enumerate(splits):
        progress(f"Stage 2 Q1/Q2: held-out fold {fold + 1}/6")
        model = Pipeline([("prep", preprocessing()), ("model", MultiOutputRegressor(GradientBoostingRegressor(n_estimators=ext["gbr_estimators"], learning_rate=ext["gbr_learning_rate"], max_depth=ext["gbr_depth"], min_samples_leaf=5, max_features=.7, random_state=cfg["seed"]), n_jobs=2))])
        # Threads avoid process resource-tracker failures on Unicode Windows temp paths.
        with joblib.parallel_backend("threading"):
            model.fit(X[train], Y[train])
            prediction = model.predict(X[test])
        sums[test] += prediction
        counts[test] += 1
        assignments.extend({"repeat": fold // 3, "fold": fold % 3, "beer": int(data.meta.iloc[i].beer)} for i in test)
        ridge, _ = fitted_model("ridge", X[train], Y[train], cfg)
        ridge_pred = ridge.predict(X[test])
        grades = assumption_grades(Y[train], ridge_pred)
        grade_min[test] = np.minimum(grade_min[test], grades.min(axis=0))
        grade_max[test] = np.maximum(grade_max[test], grades.max(axis=0))
        grade_disagreement.append({"fold": fold, "any_rule_disagreement_fraction": float((grades.max(0) != grades.min(0)).mean()), "mean_grade_spread": float((grades.max(0) - grades.min(0)).mean())})
        # Exact linear predictor permutation. The trained preprocessing is frozen.
        Z = ridge.named_steps["prep"].transform(X[test])
        coefficients = ridge.named_steps["model"].coef_
        baseline_error = np.abs(Y[test] - ridge_pred).mean(axis=0)
        rng = np.random.default_rng(cfg["seed"] + fold)
        deltas = np.zeros_like(permutation_sums)
        for j in range(X.shape[1]):
            for _ in range(ext["permutation_repeats"]):
                permuted = Z[rng.permutation(len(Z)), j] - Z[:, j]
                pp = ridge_pred + permuted[:, None] * coefficients[:, j]
                deltas[j] += np.abs(Y[test] - pp).mean(axis=0) - baseline_error
        deltas /= ext["permutation_repeats"]
        permutation_sums += deltas
        positive_folds += deltas > 0
    gbr = sums / counts[:, None]
    existing = pd.read_csv(base / "q1/metrics_by_target.csv")
    additional = pd.DataFrame(metric_rows(Y, gbr, targets, "gbr_fixed"))
    ridge_oof = pd.read_csv(base / "q1/oof_ridge_continuous.csv")[targets].to_numpy()
    tree_oof = pd.read_csv(base / "q1/oof_extra_trees_continuous.csv")[targets].to_numpy()
    blend = (ridge_oof + tree_oof) / 2
    additional = pd.concat([additional, pd.DataFrame(metric_rows(Y, blend, targets, "fixed_ridge_tree_blend"))], ignore_index=True)
    all_metrics = pd.concat([existing, additional], ignore_index=True)
    save_csv(all_metrics, out / "q1/metrics_by_target.csv")
    save_csv(all_metrics.groupby("model")[["mae", "rmse", "r2"]].mean().reset_index(), out / "q1/metrics_summary.csv")
    save_csv(pd.concat([data.meta[["beer"]], pd.DataFrame(gbr, columns=targets)], axis=1), out / "q1/oof_gbr_continuous.csv")
    save_csv(pd.DataFrame(assignments), out / "q1/split_assignments.csv")
    save_csv(pd.DataFrame(grade_disagreement), out / "q1/grade_rule_sensitivity.csv")
    grade_ranges = pd.concat([data.meta[["beer"]], pd.DataFrame(grade_min.astype(int), columns=[c + "__min" for c in targets]), pd.DataFrame(grade_max.astype(int), columns=[c + "__max" for c in targets])], axis=1)
    grade_ranges["status"] = "range_over_assumed_rules_and_CV_fits_not_a_confidence_interval"
    save_csv(grade_ranges, out / "q1/ASSUMPTION_grade_ranges.csv")

    # Proper train / calibration / test split for one demonstrative conformal experiment.
    train, rest = train_test_split(np.arange(len(X)), test_size=.4, random_state=cfg["seed"] + 71)
    calibration, test = train_test_split(rest, test_size=.5, random_state=cfg["seed"] + 72)
    ridge, _ = fitted_model("ridge", X[train], Y[train], cfg)
    q = conformal_quantile(np.abs(Y[calibration] - ridge.predict(X[calibration])), ext["conformal_alpha"])
    pp = ridge.predict(X[test])
    coverage = (np.abs(Y[test] - pp) <= q).mean(axis=0)
    save_csv(pd.DataFrame({"target": targets, "nominal_marginal_coverage": .9, "empirical_test_coverage": coverage, "interval_width": 2 * q, "calibration_n": len(calibration), "test_n": len(test)}), out / "q1/conformal_coverage.csv")
    save_csv(pd.DataFrame({"beer": data.meta.beer, "role": ["train" if i in set(train) else "calibration" if i in set(calibration) else "test" for i in range(len(X))]}), out / "q1/conformal_split.csv")

    progress("Stage 2 Q2: assembling held-out attribution evidence")
    importances = permutation_sums / len(splits)
    rows = [{"target": target, "field": field, "heldout_permutation_MAE_increase": importances[j, k], "positive_fold_fraction": positive_folds[j, k] / len(splits)} for k, target in enumerate(targets) for j, field in enumerate(features)]
    importance_frame = pd.DataFrame(rows)
    save_csv(importance_frame, out / "q2/heldout_permutation_importance.csv")
    top = pd.read_csv(base / "q2/top3_by_sensory_dimension.csv").merge(importance_frame, on=["target", "field"], validate="one_to_one")
    top["evidence_status"] = np.where((top.bootstrap_top3_frequency >= .5) & (top.heldout_permutation_MAE_increase > 0) & (top.positive_fold_fraction >= 2 / 3), "comparatively_supported_association", "weak_or_unstable_association")
    top["causality"] = "not_established"
    save_csv(top, out / "q2/top3_evidence_review.csv")

    progress("Stage 2 Q3: comparing conditional generators across held-out splits")
    rng = np.random.default_rng(cfg["seed"])
    results = []
    final_generator, final_classifier, generated_final = None, None, []
    first_train = None
    for repeat in range(ext["generation_repeats"]):
        train, test = train_test_split(np.arange(len(X)), test_size=.2, stratify=labels, random_state=cfg["seed"] + repeat)
        generator = ConditionalPCAGaussian(10, cfg["seed"]).fit(X[train], labels[train])
        clf = Pipeline([("prep", preprocessing()), ("model", LogisticRegression(C=1, max_iter=2000, class_weight="balanced", random_state=cfg["seed"]))]).fit(X[train], labels[train])
        acc = balanced_accuracy_score(labels[test], clf.predict(X[test]))
        for style in ext["generation_styles"]:
            real = X[test[labels[test] == style]]
            real_train = generator.prep.named_steps["scale"].inverse_transform(generator.prep.transform(X[train[labels[train] == style]]))
            samples = {"bootstrap": real_train[rng.integers(0, len(real_train), ext["generation_n"])], "local_convex": local_convex_samples(real_train, ext["generation_n"], rng), "PCA_Gaussian": generator.sample(style, ext["generation_n"], rng)}
            for method, values in samples.items():
                Z = generator.prep.transform(values)
                reference = generator.prep.transform(real)
                distances = cdist(Z, generator.prep.transform(X[train]))
                ks = [ks_2samp(values[:, j], real[:, j][np.isfinite(real[:, j])]).statistic for j in range(X.shape[1]) if np.isfinite(real[:, j]).any()]
                results.append({"repeat": repeat, "style": style, "method": method, "n_real_holdout": len(real), "KS_mean": np.mean(ks), "MMD2": biased_mmd(Z, reference), "exact_training_copy_fraction": float((distances.min(1) < 1e-8).mean()), "target_style_probability": clf.predict_proba(values)[:, list(clf.classes_).index(style)].mean(), "style_classifier_holdout_balanced_accuracy": acc})
                if repeat == 0 and method == "local_convex":
                    frame = pd.DataFrame(values, columns=features)
                    frame.insert(0, "style", style)
                    frame.insert(0, "generated_id", [f"LC_{style}_{j+1:03d}" for j in range(len(values))])
                    generated_final.append(frame)
        if repeat == 0:
            final_generator, final_classifier, first_train = generator, clf, train
    diagnostics = pd.DataFrame(results)
    save_csv(diagnostics, out / "q3/generator_comparison_by_split.csv")
    save_csv(diagnostics.groupby(["method", "style"])[["KS_mean", "MMD2", "exact_training_copy_fraction", "target_style_probability"]].mean().reset_index(), out / "q3/generator_comparison_mean.csv")
    generated = pd.concat(generated_final, ignore_index=True)
    save_csv(generated, out / "q3/local_convex_generated_profiles.csv")
    bundle = joblib.load(base / "q1/private/model.joblib")
    predictor = bundle["model"]
    portraits = pd.concat([generated[["generated_id", "style"]], pd.DataFrame(predictor.predict(generated[features].to_numpy()), columns=targets)], axis=1)
    save_csv(portraits, out / "q3/local_convex_sensory_continuous.csv")
    transfer_results = expanded_transfer(data, final_generator, final_classifier, first_train, predictor, ext)
    save_csv(transfer_results[0], out / "q3/expanded_transfer_diagnostics.csv")
    save_csv(transfer_results[1], out / "q3/expanded_transfer_profiles.csv")
    save_csv(transfer_results[2], out / "q3/expanded_transfer_defect_checks.csv")
    write_json(out / "status.json", {
        "absolute_score_calibration": "not_identifiable_from_provided_aggregated_per_taster_standardized_scores_without_additional_anchors",
        "grade_sensitivity_disagreement": float(pd.DataFrame(grade_disagreement).any_rule_disagreement_fraction.mean()),
        "conformal_nominal_marginal_coverage": .9, "conformal_empirical_average_coverage": float(coverage.mean()),
        "conformal_caveat": "single random split; marginal coverage on provided continuous labels, not simultaneous or original-scale coverage",
        "comparatively_supported_top3_associations": int((top.evidence_status == "comparatively_supported_association").sum()),
        "top3_total": len(top), "generation_repeated_splits": ext["generation_repeats"],
        "expanded_transfer_nontrivial": int((transfer_results[0].status == "nontrivial_model_feasible_candidate").sum()),
        "physical_or_tasting_validation": "not_performed", "research_stage": "exploratory_extension_after_stage_one_not_an_untouched_external_test",
    })


def expanded_transfer(data, generator, classifier, train, predictor, ext):
    X = data.X.to_numpy()
    features, targets = data.X.columns.tolist(), data.Y.columns.tolist()
    labels = data.meta.tasting_category_fine.to_numpy()
    positions = [features.index(c) for c in CHEMICAL_DEFECTS]
    stale = targets.index("stale_hops")
    target_train = X[train[labels[train] == "Tripel"]]
    target_train = generator.prep.named_steps["scale"].inverse_transform(generator.prep.transform(target_train))
    target_class = list(classifier.classes_).index("Tripel")
    diagnostics, profiles, checks = [], [], []
    for i in np.flatnonzero(labels == "Pils/Lager"):
        x = np.where(np.isfinite(X[i]), X[i], generator.median)
        source_pred = predictor.predict(x[None])[0]
        base_prob = classifier.predict_proba(x[None])[0, target_class]
        cleaned = np.clip(x, generator.low, generator.high)
        cleaned[positions] = np.minimum(cleaned[positions], x[positions])
        clean_prob = classifier.predict_proba(cleaned[None])[0, target_class]
        candidates = []
        for target in target_train:
            for weight in np.linspace(.025, 1, ext["transfer_weights"]):
                value = np.clip(x + weight * (target - x), generator.low, generator.high)
                value[positions] = np.minimum(value[positions], x[positions])
                candidates.append(value)
        candidates = np.asarray(candidates)
        pp = predictor.predict(candidates)
        probabilities = classifier.predict_proba(candidates)[:, target_class]
        valid = chemical_nonincrease(x, candidates, positions) & (pp[:, stale] <= source_pred[stale] + 1e-9) & (probabilities > max(base_prob, clean_prob) + .01)
        if valid.any():
            ids = np.flatnonzero(valid)
            j = ids[np.argmax(probabilities[ids])]
            value, portrait, prob, status = candidates[j], pp[j], probabilities[j], "nontrivial_model_feasible_candidate"
        else:
            value, portrait, prob, status = x, source_pred, base_prob, "no_nontrivial_candidate_found_source_retained"
        beer = int(data.meta.iloc[i].beer)
        missing = [c for c, j in zip(CHEMICAL_DEFECTS, positions) if not np.isfinite(X[i, j])]
        diagnostics.append({"beer": beer, "status": status, "generator_role": "train" if i in set(train) else "holdout", "source_trip_probability": base_prob, "candidate_trip_probability": prob, "cleaning_only_probability": clean_prob, "change_l2": np.linalg.norm(value - x), "missing_source_defects": ",".join(missing), "all_source_defects_observed": not missing, "stale_constraint": "predicted_proxy_only"})
        profiles.append({"beer": beer, **dict(zip(features, value))})
        for name, j in zip(CHEMICAL_DEFECTS, positions):
            checks.append({"beer": beer, "defect": name, "source": x[j], "candidate": value[j], "nonincrease": bool(value[j] <= x[j] + 1e-9), "source_observed": bool(np.isfinite(X[i, j])), "scale": "provided_measurement"})
        checks.append({"beer": beer, "defect": "stale_hops", "source": source_pred[stale], "candidate": portrait[stale], "nonincrease": bool(portrait[stale] <= source_pred[stale] + 1e-9), "source_observed": False, "scale": "learned_sensory_proxy"})
    return pd.DataFrame(diagnostics), pd.DataFrame(profiles), pd.DataFrame(checks)
