"""Stage-three algorithm prototypes; AI-assisted, OpenAI Codex, 2026-10-25.

These helpers validate mathematical properties. They are not evidence that the
proposed full study improves real prediction or chemical feasibility.
"""
import numpy as np


def fixed_grade_boundaries(training_targets):
    """Fit once on training labels; never refit with calibration/test labels."""
    y = np.asarray(training_targets, float)
    if y.ndim != 2 or not np.isfinite(y).all() or len(y) < 2:
        raise ValueError("Expected at least two finite training rows")
    return np.quantile(y, np.arange(1, 7) / 7, axis=0)


def grade_with_fixed_boundaries(values, boundaries):
    values, boundaries = np.asarray(values, float), np.asarray(boundaries, float)
    if values.ndim != 2 or boundaries.shape != (6, values.shape[1]):
        raise ValueError("Grade dimensions do not agree")
    if not np.isfinite(values).all() or not np.isfinite(boundaries).all() or np.any(np.diff(boundaries, axis=0) < 0):
        raise ValueError("Need finite values and ordered boundaries")
    return (values[:, None, :] > boundaries[None, :, :]).sum(axis=1).astype(int)


def ordinal_interval_image(lower, upper, boundaries):
    """Conservative image of a continuous interval under a fixed monotone map.

    Coverage, if applicable, is for this operationally defined grade only.
    Tied boundaries can make the interval hull include unreachable categories.
    """
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if lower.shape != upper.shape or np.any(lower > upper):
        raise ValueError("Invalid continuous intervals")
    return grade_with_fixed_boundaries(lower, boundaries), grade_with_fixed_boundaries(upper, boundaries)


def shrunk_blend_weights(y_inner_oof, et_inner_oof, gbr_inner_oof, penalty):
    """Fit using inner OOF predictions only; choose penalty within inner training.

    w[j] multiplies GBR and (1-w[j]) multiplies ExtraTrees. The closed-form
    objective is mean squared error plus penalty*(w[j]-global_weight)**2.
    """
    y, et, gbr = [np.asarray(a, float) for a in (y_inner_oof, et_inner_oof, gbr_inner_oof)]
    if y.ndim != 2 or et.shape != y.shape or gbr.shape != y.shape or penalty < 0:
        raise ValueError("Invalid blend inputs")
    if not all(np.isfinite(a).all() for a in (y, et, gbr)):
        raise ValueError("Blend arrays must be finite")
    diff, residual = gbr - et, y - et
    denom = np.mean(diff * diff)
    global_weight = float(np.clip(np.mean(diff * residual) / denom, 0, 1)) if denom > 1e-14 else 0.5
    local_denom = np.mean(diff * diff, axis=0) + penalty
    numerator = np.mean(diff * residual, axis=0) + penalty * global_weight
    weights = np.divide(numerator, local_denom, out=np.full(y.shape[1], global_weight), where=local_denom > 1e-14)
    return np.clip(weights, 0, 1), global_weight


def paired_oof_bootstrap(y, pred_a, pred_b, strata, repeats=2000, seed=20261025):
    """Stratified beer-level resampling of fixed predictions; NOT a retraining CI.

    Inputs must have one row per unique beer, with repeats already averaged.
    Negative differences favour A. Macro MAE is average over rows and targets.
    """
    y, a, b = [np.asarray(x, float) for x in (y, pred_a, pred_b)]
    labels = np.asarray(strata)
    if y.ndim != 2 or a.shape != y.shape or b.shape != y.shape or len(labels) != len(y):
        raise ValueError("Paired dimensions differ")
    if repeats < 1 or not all(np.isfinite(x).all() for x in (y, a, b)):
        raise ValueError("Need finite paired predictions and positive repeats")
    per_beer = (np.abs(y-a)-np.abs(y-b)).mean(axis=1)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    rng = np.random.default_rng(seed)
    draws = np.array([np.mean(per_beer[np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])]) for _ in range(repeats)])
    return {"difference_A_minus_B": float(per_beer.mean()), "conditional_percentile_95": np.quantile(draws, [.025, .975]).tolist(),
            "scope": "conditional_on_fixed_OOF_predictions_does_not_include_model_retraining_or_selection_uncertainty",
            "n_unique_beers": len(y), "resamples": repeats}


def pareto_minimize(objectives):
    """Keep non-dominated finite candidates; no weighting can hide a violation."""
    values = np.asarray(objectives, float)
    if values.ndim != 2:
        raise ValueError("Expected candidates by objectives")
    keep = np.isfinite(values).all(axis=1)
    for i in np.flatnonzero(keep):
        dominated = np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1) & np.isfinite(values).all(axis=1)
        if dominated.any():
            keep[i] = False
    return keep


def threshold_excess(concentration, threshold):
    """Use only after unit and scale verification; invalid inputs stay unknown."""
    concentration, threshold = np.broadcast_arrays(np.asarray(concentration, float), np.asarray(threshold, float))
    valid = np.isfinite(concentration) & np.isfinite(threshold) & (concentration >= 0) & (threshold > 0)
    out = np.full(concentration.shape, np.nan)
    out[valid] = np.maximum(concentration[valid] / threshold[valid] - 1, 0)
    return out
