"""Power planning and analysis for a future paired, blinded physical experiment.
No human observations are generated here. AI-assisted, 2026-10-25.
"""
import numpy as np
import pandas as pd
from scipy.stats import t, nct

DEFECT_ENDPOINTS = ['dms', 'diacetyl', 'X4vg', 't2n', 'stale_hops']


def planned_noninferiority_power(n, sigma, margin=.5, alpha=.01):
    critical = t.ppf(1 - alpha, n - 1)
    return float(nct.cdf(-critical, n - 1, -margin * np.sqrt(n) / sigma))


def analyze_paired_trial(frame, margin=.5):
    endpoints = DEFECT_ENDPOINTS + ['style_Tripel_fit']
    required = ['panelist', 'batch', 'condition'] + endpoints
    if not set(required).issubset(frame) or frame.empty:
        raise ValueError('No observed paired data; a blank template is not an experiment')
    if frame[required].isna().any().any() or frame.duplicated(['panelist', 'batch', 'condition']).any():
        raise ValueError('Missing values or duplicated participant-batch-condition records require review')
    if set(frame.condition) != {'control', 'candidate'} or not frame[endpoints].ge(0).all().all() or not frame[endpoints].le(6).all().all():
        raise ValueError('Two conditions and valid 0-6 ratings are required')
    control = frame[frame.condition == 'control'].set_index(['panelist', 'batch'])[endpoints]
    candidate = frame[frame.condition == 'candidate'].set_index(['panelist', 'batch'])[endpoints]
    if set(control.index) != set(candidate.index):
        raise ValueError('Unpaired batches cannot silently enter paired analysis')
    difference = (candidate - control).groupby(level='panelist').mean()
    if len(difference) < 3:
        raise ValueError('At least three independent panelists are required for the diagnostic')
    rows = []
    for endpoint in endpoints:
        d = difference[endpoint].to_numpy(); n = len(d); mean = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
        alpha = .01 if endpoint in DEFECT_ENDPOINTS else .05
        upper = mean + t.ppf(1 - alpha, n - 1) * se
        lower = mean - t.ppf(1 - alpha, n - 1) * se
        rows.append({'endpoint': endpoint, 'independent_panelists': n, 'candidate_minus_control': mean,
                     'one_sided_alpha': alpha, 'lower_confidence_limit': lower, 'upper_confidence_limit': upper,
                     'criterion_passed': bool(upper < margin if endpoint in DEFECT_ENDPOINTS else lower > 0),
                     'criterion': 'upper_limit_below_0.5_noninferiority_margin' if endpoint in DEFECT_ENDPOINTS else 'lower_limit_above_zero_style_improvement'})
    return pd.DataFrame(rows)
