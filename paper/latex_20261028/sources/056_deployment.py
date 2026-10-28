"""Final full-data models and label-free inference for the historical panel.

Only load joblib files from a trusted source: deserialization executes code.
"""
from pathlib import Path
import csv
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd

from .common import sha256
from .data import META
from .stage3_prediction import fit_pair, predict_pair, tree_model


def read_input(path, features):
    """Require the complete training schema; reorder columns by recorded names."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError('Input CSV contains duplicate column names')
    data = pd.read_csv(path)
    missing = [name for name in features if name not in data.columns]
    if missing:
        raise ValueError(f'Missing {len(missing)} required feature(s): {missing[:12]}')
    if data.empty:
        raise ValueError('Input CSV contains no samples')
    numeric = data.loc[:, features].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError('Infinite feature values are not supported')
    if 'beer' in data and (data.beer.isna().any() or data.beer.duplicated().any()):
        raise ValueError('beer identifiers must be non-null and unique')
    return data, numeric


def fit_final(chemical_path, means_path, distribution_path, cfg, unit_map=None, scale_map=None):
    """Fit the same half ET/GBR mean and frequency ET models on all available beers."""
    chemical = pd.read_csv(chemical_path)
    if 'beer' not in chemical or chemical.beer.isna().any() or chemical.beer.duplicated().any():
        raise ValueError('Training chemical data need unique beer identifiers')
    features = [c for c in chemical.columns if c not in META]
    _, X = read_input(chemical_path, features)
    means = pd.read_csv(means_path)
    if means.beer.isna().any() or means.beer.duplicated().any() or set(means.beer) != set(chemical.beer):
        raise ValueError('Training means must match all chemical beer identifiers exactly')
    targets = [c for c in means if c != 'beer']
    if not targets:
        raise ValueError('No sensory targets found')
    Y = means.set_index('beer').loc[chemical.beer, targets].to_numpy(dtype=float)
    if not np.isfinite(Y).all() or (Y < 0).any() or (Y > 6).any():
        raise ValueError('Panel means must be finite and in [0,6]')
    q_frame = pd.read_csv(distribution_path)
    q_cols = [f'observed_panel_p{k}' for k in range(7)]
    if q_frame.duplicated(['beer', 'target']).any():
        raise ValueError('Observed distribution has duplicate beer/target records')
    idx = pd.MultiIndex.from_product([chemical.beer, targets], names=['beer', 'target'])
    q_frame = q_frame.set_index(['beer', 'target'])
    if len(q_frame) != len(idx) or not idx.isin(q_frame.index).all():
        raise ValueError('Observed distributions must match all beers and targets')
    q = q_frame.loc[idx, q_cols].to_numpy(dtype=float).reshape(len(X), len(targets), 7)
    if not np.isfinite(q).all() or (q < -1e-8).any() or not np.allclose(q.sum(-1), 1, atol=1e-6):
        raise ValueError('Observed ratings must form valid probability distributions')
    q = np.maximum(q, 0); q /= q.sum(-1, keepdims=True)
    et, gbr = fit_pair(X, Y, cfg)
    ordinal = tree_model(cfg).fit(X, q.reshape(len(X), -1))
    descriptions = {name: {'field': name, 'reference_concentration_unit': None,
                           'empirical_transform': None, 'transform_match_fraction': None}
                    for name in features}
    for path, columns in [(unit_map, {'unit': 'reference_concentration_unit'}),
                          (scale_map, {'best_transform': 'empirical_transform', 'match_fraction': 'transform_match_fraction'})]:
        if path is not None:
            table = pd.read_csv(path).drop_duplicates('field').set_index('field')
            for field in set(features) & set(table.index):
                for original, dest in columns.items():
                    value = table.at[field, original]
                    descriptions[field][dest] = None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value
    manifest = {
        'format_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'features': features, 'targets': targets, 'training_beers': len(X),
        'feature_descriptions': [descriptions[name] for name in features],
        'feature_scale': 'same field names, units and transformations as the official attachment 1.csv; do not substitute raw concentration for log-scale input',
        'mean_endpoint': 'equal-person panel mean on 0-6 scale; ET and GBR fixed equal-weight blend',
        'grade_endpoint': 'mode and median of predicted seven-category individual-rating distribution',
        'probability_calibration': 'not independently established',
        'model_assessment': 'full-data fit, not an independent accuracy evaluation',
        'training_hashes_sha256': {Path(p).name: sha256(p) for p in (chemical_path, means_path, distribution_path, unit_map, scale_map) if p is not None},
        'versions': {'python': platform.python_version(), **{p: version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'joblib')}},
        'config': cfg,
    }
    return {'manifest': manifest, 'mean_et': et, 'mean_gbr': gbr, 'ordinal_et': ordinal}


def predict(bundle, input_path):
    metadata = bundle['manifest']; data, X = read_input(input_path, metadata['features'])
    pe, pg = predict_pair((bundle['mean_et'], bundle['mean_gbr']), X)
    pe = np.asarray(pe).reshape(len(X), len(metadata['targets']))
    pg = np.asarray(pg).reshape(len(X), len(metadata['targets']))
    means = np.clip((pe + pg) / 2, 0, 6)
    p = bundle['ordinal_et'].predict(X).reshape(len(X), len(metadata['targets']), 7)
    if not np.isfinite(p).all() or (p < -1e-8).any() or not np.allclose(p.sum(-1), 1, atol=1e-6):
        raise ValueError('Model produced invalid rating probabilities')
    p = np.maximum(p, 0); p /= p.sum(-1, keepdims=True)
    ids = data['beer'].to_numpy() if 'beer' in data else np.arange(1, len(data) + 1)
    result = {'beer': ids}
    targets = metadata['targets']
    tables = {}
    for name, values in [('panel_mean', means), ('modal_grade', p.argmax(-1)), ('median_grade', (p.cumsum(-1) >= .5).argmax(-1))]:
        tables[name] = pd.concat([pd.DataFrame(result), pd.DataFrame(values, columns=targets)], axis=1)
    records = pd.DataFrame({'beer': np.repeat(ids, len(targets)), 'target': np.tile(targets, len(ids))})
    for k in range(7): records[f'p{k}'] = p[:, :, k].ravel()
    tables['rating_distribution'] = records
    return tables


def load_trusted_model(path):
    bundle = joblib.load(path)
    if bundle.get('manifest', {}).get('format_version') != 1:
        raise ValueError('Unsupported model bundle')
    installed = version('scikit-learn')
    if installed != bundle['manifest']['versions']['scikit-learn']:
        raise ValueError(f'scikit-learn version mismatch: trained {bundle["manifest"]["versions"]["scikit-learn"]}, installed {installed}')
    return bundle
