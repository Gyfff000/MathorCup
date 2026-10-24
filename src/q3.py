"""AI-assisted implementation: OpenAI Codex / GPT-6, 2026-10-24.
Small-sample conditional Gaussian reference and explicitly model-based transfers.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .common import ecdf_grades, preprocessing, progress, save_csv, write_json
from .data import CHEMICAL_DEFECTS


class ConditionalPCAGaussian:
    def __init__(self, components=10, seed=20261024):
        self.components, self.seed = components, seed

    def fit(self, X, labels):
        self.prep = preprocessing()
        z = self.prep.fit_transform(X)
        self.pca = PCA(n_components=min(self.components, len(z) - 1, z.shape[1]), random_state=self.seed)
        latent = self.pca.fit_transform(z)
        self.low = np.nanquantile(X, .01, axis=0)
        self.high = np.nanquantile(X, .99, axis=0)
        self.median = np.nanmedian(X, axis=0)
        pooled = LedoitWolf().fit(latent).covariance_
        self.means, self.covariances = {}, {}
        labels = np.asarray(labels)
        for style in np.unique(labels):
            a = latent[labels == style]
            self.means[style] = a.mean(axis=0)
            local = np.atleast_2d(np.cov(a, rowvar=False)) if len(a) > 1 else pooled
            weight = len(a) / (len(a) + 10)
            self.covariances[style] = weight * local + (1 - weight) * pooled + 1e-8 * np.eye(pooled.shape[0])
        return self

    def encode(self, X):
        return self.pca.transform(self.prep.transform(X))

    def decode(self, latent):
        z = self.pca.inverse_transform(latent)
        values = self.prep.named_steps["scale"].inverse_transform(z)
        return np.clip(values, self.low, self.high)

    def sample(self, style, n, rng):
        latent = rng.multivariate_normal(self.means[style], self.covariances[style], size=n)
        return self.decode(latent)


def chemical_nonincrease(source, candidate, positions, tolerance=1e-9):
    return np.all(np.asarray(candidate)[..., positions] <= np.asarray(source)[..., positions] + tolerance, axis=-1)


def biased_mmd(a, b):
    """RBF MMD squared; bandwidth from real reference only."""
    if len(a) < 2 or len(b) < 2:
        return np.nan
    dist = cdist(b, b, metric="sqeuclidean")
    positive = dist[dist > 0]
    bandwidth = np.median(positive) if len(positive) else 1.
    return float(np.exp(-cdist(a, a, "sqeuclidean") / (2 * bandwidth)).mean() + np.exp(-dist / (2 * bandwidth)).mean() - 2 * np.exp(-cdist(a, b, "sqeuclidean") / (2 * bandwidth)).mean())


def run_q3(data, out, cfg):
    dest = Path(out) / "q3"
    dest.mkdir(parents=True, exist_ok=True)
    path = Path(out) / "q1/private/model.joblib"
    if not path.exists():
        raise FileNotFoundError("Q3 needs the Q1 reference predictor; run q1 first")
    bundle = joblib.load(path)
    predictor = bundle["model"]
    features, targets = data.X.columns.tolist(), data.Y.columns.tolist()
    if bundle["features"] != features or bundle["targets"] != targets:
        raise ValueError("Q1 predictor schema mismatch")
    X = data.X.to_numpy()
    labels = data.meta.tasting_category_fine.to_numpy()
    train, test = train_test_split(np.arange(len(X)), test_size=cfg["generation_holdout_fraction"], stratify=labels, random_state=cfg["seed"])
    save_csv(pd.DataFrame({"beer": data.meta.beer, "role": ["train" if i in set(train) else "holdout" for i in range(len(X))]}), dest / "generator_split.csv")
    progress("Q3: fitting conditional PCA Gaussian on generator training samples only")
    generator = ConditionalPCAGaussian(cfg["generation_components"], cfg["seed"]).fit(X[train], labels[train])
    classifier = Pipeline([("prep", preprocessing()), ("model", LogisticRegression(C=1, max_iter=2000, class_weight="balanced", random_state=cfg["seed"]))])
    classifier.fit(X[train], labels[train])
    class_index = {s: j for j, s in enumerate(classifier.classes_)}
    holdout_balanced = float(balanced_accuracy_score(labels[test], classifier.predict(X[test])))
    rng = np.random.default_rng(cfg["seed"])
    generated, metrics, portraits = [], [], []
    for style in cfg["generation_styles"]:
        values = generator.sample(style, cfg["generation_samples_per_style"], rng)
        ids = [f"{style}_{i + 1:03d}" for i in range(len(values))]
        frame = pd.DataFrame(values, columns=features)
        frame.insert(0, "style", style)
        frame.insert(0, "generated_id", ids)
        generated.append(frame)
        pred = predictor.predict(values)
        portrait = pd.DataFrame(pred, columns=targets)
        portrait.insert(0, "style", style)
        portrait.insert(0, "generated_id", ids)
        portraits.append(portrait)
        real = X[test[labels[test] == style]]
        zs = generator.prep.transform(values)
        zr = generator.prep.transform(real)
        ztrain = generator.prep.transform(X[train])
        distances = cdist(zs, ztrain)
        train_style = X[train[labels[train] == style]]
        bootstrap = train_style[rng.integers(0, len(train_style), len(values))]
        # Natural missingness in held-out measurements must not erase all KS diagnostics.
        ks_values = []
        for j in range(X.shape[1]):
            observed = real[:, j][np.isfinite(real[:, j])]
            if len(observed):
                ks_values.append(ks_2samp(values[:, j], observed, method="auto").statistic)
        # Both metrics use independent real holdout samples; small counts are explicit.
        metrics.append({"style": style, "n_generated": len(values), "n_training_style": len(train_style), "n_holdout_style": len(real),
                        "mean_marginal_KS_holdout": float(np.mean(ks_values)), "n_KS_fields_observed": len(ks_values),
                        "RBF_MMD2_holdout": biased_mmd(zs, zr),
                        "bootstrap_baseline_MMD2_holdout": biased_mmd(generator.prep.transform(bootstrap), zr),
                        "mean_target_style_probability": float(classifier.predict_proba(values)[:, class_index[style]].mean()),
                        "nearest_train_distance_mean": float(distances.min(axis=1).mean()),
                        "exact_training_copy_fraction": float((distances.min(axis=1) < 1e-8).mean()),
                        "heldout_reference_warning": "few independent style examples; descriptive distances only, no quality guarantee"})
    all_generated = pd.concat(generated, ignore_index=True)
    save_csv(all_generated, dest / "generated_compound_profiles.csv")
    save_csv(pd.concat(portraits, ignore_index=True), dest / "generated_sensory_continuous.csv")
    save_csv(pd.DataFrame(metrics), dest / "generation_diagnostics.csv")
    if cfg["export_assumption_grades"]:
        a = pd.concat(portraits, ignore_index=True)
        grade = ecdf_grades(data.Y.to_numpy()[train], a[targets].to_numpy())
        proxy = pd.concat([a[["generated_id", "style"]], pd.DataFrame(grade, columns=targets)], axis=1)
        proxy["status"] = "ASSUMPTION_ONLY_ECDF_not_absolute_sensory_scores"
        save_csv(proxy, dest / "ASSUMPTION_generated_percentile_grades.csv")
    positions = [features.index(c) for c in CHEMICAL_DEFECTS]
    stale = targets.index("stale_hops")
    target = cfg["transfer_target"]
    source = cfg["transfer_source"]
    transfers, diagnostics, checks, transfer_sensory = [], [], [], []
    for i in np.flatnonzero(labels == source):
        original = X[i]
        x = np.where(np.isfinite(original), original, generator.median)
        base_pred = predictor.predict(x[None])[0]
        base_prob = classifier.predict_proba(x[None])[0, class_index[target]]
        cleaning_only = np.clip(x, generator.low, generator.high)
        cleaning_only[positions] = np.minimum(cleaning_only[positions], x[positions])
        cleaning_prob = classifier.predict_proba(cleaning_only[None])[0, class_index[target]]
        latent = generator.encode(x[None])[0]
        desired = generator.decode((latent + generator.means[target] - generator.means[source])[None])[0]
        candidates = []
        for t in np.linspace(.025, 1.25, 50):
            candidate = np.clip(x + t * (desired - x), generator.low, generator.high)
            candidate[positions] = np.minimum(candidate[positions], x[positions])
            candidates.append(candidate)
        candidates = np.asarray(candidates)
        candidate_pred = predictor.predict(candidates)
        chemical_ok = chemical_nonincrease(x, candidates, positions)
        stale_ok = candidate_pred[:, stale] <= base_pred[stale] + 1e-9
        target_prob = classifier.predict_proba(candidates)[:, class_index[target]]
        # Separate genuine style movement from the effects of clipping source outliers.
        feasible = chemical_ok & stale_ok & (target_prob > max(base_prob, cleaning_prob) + .01)
        if feasible.any():
            eligible = np.flatnonzero(feasible)
            chosen = eligible[np.argmax(target_prob[eligible])]
            value, pp, probability = candidates[chosen], candidate_pred[chosen], target_prob[chosen]
            status = "nontrivial_model_feasible_candidate"
        else:
            value, pp, probability = x.copy(), base_pred.copy(), base_prob
            status = "no_nontrivial_candidate_found_source_retained"
        beer = int(data.meta.iloc[i].beer)
        missing_defects = [c for c, j in zip(CHEMICAL_DEFECTS, positions) if not np.isfinite(original[j])]
        for c, j in zip(CHEMICAL_DEFECTS, positions):
            checks.append({"beer": beer, "defect": c, "source_value": x[j], "candidate_value": value[j], "nonincrease": bool(value[j] <= x[j] + 1e-9), "scale": "supplied_feature_scale", "source_observed": bool(np.isfinite(original[j])), "absolute_threshold_certified": False})
        checks.append({"beer": beer, "defect": "stale_hops", "source_value": base_pred[stale], "candidate_value": pp[stale], "nonincrease": bool(pp[stale] <= base_pred[stale] + 1e-9), "scale": "Q1_model_predicted_continuous_sensory", "source_observed": False, "absolute_threshold_certified": False})
        transfers.append({"beer": beer, "source_style": source, "target_style": target, **dict(zip(features, value))})
        transfer_sensory.append({"beer": beer, **dict(zip(targets, pp))})
        diagnostics.append({"beer": beer, "generator_role": "holdout" if i in set(test) else "train", "status": status,
                            "source_target_probability": base_prob, "candidate_target_probability": probability,
                            "cleaning_only_target_probability": cleaning_prob,
                            "supplied_scale_change_l2": float(np.linalg.norm(value - x)),
                            "chemical_nonincrease": bool(chemical_nonincrease(x, value, positions)),
                            "predicted_stale_hops_nonincrease": bool(pp[stale] <= base_pred[stale] + 1e-9),
                            "missing_source_defects": ",".join(missing_defects),
                            "out_of_training_envelope_fields": int(((value < generator.low - 1e-9) | (value > generator.high + 1e-9)).sum()),
                            "guarantee_scope": "measurement nonincrease plus learned stale proxy; physical and sensory experiment not performed"})
    save_csv(pd.DataFrame(transfers), dest / "pils_to_tripel_profiles.csv")
    save_csv(pd.DataFrame(transfer_sensory), dest / "pils_to_tripel_sensory_continuous.csv")
    save_csv(pd.DataFrame(diagnostics), dest / "transfer_diagnostics.csv")
    save_csv(pd.DataFrame(checks), dest / "defect_nonincrease_checks.csv")
    write_json(dest / "status.json", {
        "model": "conditional_PCA_Gaussian_with_partial_covariance_pooling",
        "generator_train_n": len(train), "generator_holdout_n": len(test),
        "style_classifier_holdout_balanced_accuracy": holdout_balanced,
        "primary_feature_scale": "supplied_mixed_transforms; see audit/field_scales.csv",
        "marginal_bounds": "training 1%-99% descriptor range; not a thermodynamic or brewing-feasibility guarantee",
        "aggregate_consistency": "aggregate descriptors generated jointly; exact chemical sum identities not established",
        "sensory_portraits": "Q1 full-data fitted predictor, model-based only; not an independent tasting validation",
        "defect_thresholds": "absolute thresholds incomplete and some field scales unresolved; no complete OAV certification",
        "stale_hops_constraint": "prediction proxy only; actual sensory non-worsening unverified",
        "zero_change_transfers": "explicitly reported as no nontrivial feasible candidate, never counted as successful style transfer",
    })
