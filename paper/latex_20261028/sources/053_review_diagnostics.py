"""Scoring identities and certificates used in the 2026-10-28 revision."""
import numpy as np
from scipy.optimize import linprog


def distribution_scores(observed, predicted):
    """Return per-beer/per-attribute distances and expected proper scores."""
    q, p = np.asarray(observed), np.asarray(predicted)
    if q.shape != p.shape or q.shape[-1] != 7:
        raise ValueError('Expected matching seven-category distributions')
    for a in (q, p):
        if not np.isfinite(a).all() or a.min() < -1e-8 or not np.allclose(a.sum(-1), 1, atol=1e-7):
            raise ValueError('Invalid probability distribution')
    Q, P = q.cumsum(-1)[..., :-1], p.cumsum(-1)[..., :-1]
    distance = ((P - Q) ** 2).mean(-1)
    uncertainty = (Q * (1 - Q)).mean(-1)
    brier_distance = ((p - q) ** 2).sum(-1)
    return {'cdf_distance': distance, 'panel_disagreement': uncertainty,
            'expected_individual_RPS': distance + uncertainty,
            'brier_distance': brier_distance,
            'expected_individual_Brier': brier_distance + 1 - (q ** 2).sum(-1)}


def cdf_to_probabilities(cdf):
    a = np.asarray(cdf)
    if a.shape[-1] != 6 or not np.isfinite(a).all() or (np.diff(a, axis=-1) < -1e-10).any():
        raise ValueError('Six finite, monotone cumulative probabilities required')
    if a.min() < -1e-10 or a.max() > 1 + 1e-10:
        raise ValueError('CDF outside [0,1]')
    p = np.diff(np.concatenate([np.zeros_like(a[..., :1]), a, np.ones_like(a[..., :1])], axis=-1), axis=-1)
    p = np.maximum(p, 0)
    return p / p.sum(-1, keepdims=True)


def conditional_paired_interval(difference, strata, seed=20261028, resamples=5000):
    """Stratified beer bootstrap conditional on fixed OOF predictions."""
    d, strata = np.asarray(difference), np.asarray(strata)
    if d.ndim != 1 or len(d) != len(strata) or not np.isfinite(d).all():
        raise ValueError('One finite paired difference per beer required')
    rng = np.random.default_rng(seed)
    values = np.zeros(resamples)
    for label in np.unique(strata):
        v = d[strata == label]
        values += v[rng.integers(len(v), size=(resamples, len(v)))].sum(1)
    lo, hi = np.quantile(values / len(d), [.025, .975])
    return {'difference_A_minus_B': float(d.mean()), 'conditional_low95': float(lo),
            'conditional_high95': float(hi), 'unique_beers': len(d), 'resamples': resamples}


def mixture_certificate(difference):
    """D w <= 0, w>=0: exhibit v>=0 with D.T v >0 to exclude w!=0.

    Row normalization preserves inequalities and avoids mixing concentration units.
    Maximize the minimum normalized anchor excess over v on the simplex.
    """
    d = np.asarray(difference, dtype=float)
    if d.ndim != 2 or not d.size or not np.isfinite(d).all():
        raise ValueError('Finite defect-by-anchor differences required')
    scales = np.max(np.abs(d), axis=1)
    a = d / np.where(scales > 0, scales, 1)[:, None]
    m, n = a.shape
    result = linprog(np.r_[np.zeros(m), -1.],
                     A_ub=np.column_stack([-a.T, np.ones(n)]), b_ub=np.zeros(n),
                     A_eq=[np.r_[np.ones(m), 0]], b_eq=[1],
                     bounds=[(0, None)] * m + [(None, None)], method='highs')
    if not result.success:
        raise RuntimeError(result.message)
    v = result.x[:-1]
    margin = float((a.T @ v).min())
    return {'certified_zero_only': margin > 1e-8, 'margin': margin,
            'weights': v.tolist(), 'row_scales': scales.tolist(),
            'single_row_blockers': np.flatnonzero(d.min(1) > 1e-10).tolist()}
