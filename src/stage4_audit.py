"""Source reconciliation and raw sensory endpoints. AI-assisted, 2026-10-25.
Identity matching is provenance analysis, never chemical-model fitting.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from .common import save_csv, write_json, sha256
from .data import META, CHEMICAL_DEFECTS

ALIASES = {'A_ester_fruity': 'A_esters_fruity', 'F_ester_fruity': 'F_esters_fruity'}
RAW_NAME = 'Raw sensory data anonymized.csv'
BOOK_NAME = 'Supplemental Files and Figure source files.xlsx'


def aggregate_panel(raw, targets):
    """Repeated ratings from one person do not give that person extra weight."""
    person = raw.groupby(['beer', 'taster'])[targets].mean()
    return person.groupby('beer').mean(), raw.groupby('beer')[targets].mean(), person


def standardize_panel(raw, targets):
    group = raw.groupby('taster')[targets]
    sd = group.transform('std')
    return (raw[targets] - group.transform('mean')) / sd.replace(0, np.nan)


def panel_rating_distribution(raw, targets, beers):
    """Equal people, equal repeats per person; preserve actual integer categories."""
    encoded = np.eye(7)[raw[targets].to_numpy(int)].reshape(len(raw), -1)
    table = pd.DataFrame(encoded)
    per_person = table.groupby([raw.beer, raw.taster]).mean()
    return per_person.groupby(level=0).mean().reindex(beers).to_numpy().reshape(len(beers), len(targets), 7)


def infer_identity(standardized, raw_ids, published, discovery):
    means = standardized.groupby(raw_ids).mean().sort_index()
    distance = cdist(means[discovery], published[discovery]) / np.sqrt(len(discovery))
    i, j = linear_sum_assignment(distance)
    ordered = np.sort(distance, axis=1)
    return means, pd.DataFrame({'raw_beer': means.index.to_numpy()[i],
                               'beer': published.index.to_numpy()[j],
                               'discovery_rmse': distance[i, j],
                               'nearest_gap': ordered[:, 1] - ordered[:, 0],
                               'is_row_nearest': j == distance.argmin(1)})


def transform_candidates(raw):
    finite_positive = raw[np.isfinite(raw) & (raw > 0)]
    half_min = finite_positive.min() / 2 if len(finite_positive) else np.nan
    replacement = np.where(raw <= 0, half_min, raw)
    with np.errstate(invalid='ignore', divide='ignore'):
        return {'identity': raw, 'nonpositive_half_min': replacement,
                'log10_nonpositive_half_min': np.log10(replacement)}, half_min


def concentration_fields(calibrated, thresholds, units):
    metadata = thresholds.dropna(subset=['field_name_R']).drop_duplicates('field_name_R').set_index('field_name_R')
    fields = []
    for f in calibrated.columns:
        if f in META or f not in metadata.index or f == 'kcalperc':
            continue
        row = metadata.loc[f]
        if str(row.calibrated).startswith('yes') and row.unit in units:
            fields.append(f)
    return fields


def audit_aggregate_hypotheses(data, calibrated, out):
    """Reject unsupported aggregate identities on a disjoint row half.
    Fits are diagnostic hypotheses, not recovered author formulas.
    """
    x = data.X
    groups = {
        'esters_sum': [f for f in x if any(s in f for s in ['acetate', 'butyrate', 'hexanoate', 'octanoate', 'decanoate', 'propionate', 'isovalerate'])],
        'aroma_hops_sum': data.chemical_meta.loc[data.chemical_meta.origin_paper.astype(str).str.contains('Hops'), 'field'].tolist(),
        'acids_sum': ['acetic_acid.g.L.', 'lactic_acid.mg.L.'],
        'sulfur_sum': ['CS2', 'DES', 'DMS', 'H2S', 'MeSH', 'SO2.mg.L.']}
    rows = []
    for target, fields in groups.items():
        a = x[fields].to_numpy(float)
        probes = {'mean_processed': np.nanmean(a, axis=1), 'sum_processed': np.nansum(a, axis=1),
                  'log10_sum_exponentiated_processed': logsumexp(np.where(np.isfinite(a), a, -np.inf) * np.log(10), axis=1) / np.log(10)}
        y = x[target].to_numpy()
        for name, value in probes.items():
            valid = np.isfinite(value) & np.isfinite(y); train = valid & (np.arange(len(y)) % 2 == 0); test = valid & ~train
            slope, intercept = np.linalg.lstsq(np.c_[value[train], np.ones(train.sum())], y[train], rcond=None)[0]
            error = np.abs(value[test] * slope + intercept - y[test])
            rows.append({'aggregate': target, 'hypothesis': name, 'component_count': len(fields), 'fitted_slope_on_discovery': slope,
                         'fitted_intercept_on_discovery': intercept, 'validation_mae': error.mean(), 'validation_max_error': error.max(),
                         'identity_verified': bool(error.max() < 1e-7), 'scope': 'exploratory_formula_hypothesis_not_author_documentation'})
    features = ['ethanol..v.v.', 'GFS', 'unfermentables', 'protein.g.L.', 'glycerol']
    a = calibrated[features].to_numpy(float); y = calibrated.kcalperc.to_numpy(float)
    valid = np.isfinite(a).all(1) & np.isfinite(y); train = valid & (np.arange(len(y)) % 2 == 0); test = valid & ~train
    coef = np.linalg.lstsq(np.c_[a[train], np.ones(train.sum())], y[train], rcond=None)[0]
    error = np.abs(np.c_[a[test], np.ones(test.sum())] @ coef - y[test])
    rows.append({'aggregate': 'kcalperc', 'hypothesis': 'five_raw_nutrient_affine_fit', 'component_count': len(features),
                 'validation_mae': error.mean(), 'validation_max_error': error.max(), 'identity_verified': bool(error.max() < 1e-7),
                 'scope': 'exploratory_formula_hypothesis_not_author_documentation'})
    save_csv(pd.DataFrame(rows), Path(out) / 'aggregate_formula_hypotheses.csv')


def reconcile(data, external, out, cfg):
    external, out = Path(external), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, key in [(RAW_NAME, 'raw_md5'), (BOOK_NAME, 'workbook_md5')]:
        if hashlib.md5((external / name).read_bytes()).hexdigest() != cfg[key]:
            raise ValueError('External source checksum differs: ' + name)
    raw = pd.read_csv(external / RAW_NAME, encoding='cp1252').rename(columns=ALIASES)
    published = pd.read_excel(external / BOOK_NAME, sheet_name='Supplementary File S4')
    targets = list(data.Y)
    if set(raw.beer_name) != set(range(1, 251)) or len(published) != 250:
        raise ValueError('Unexpected source numbering; no positional fallback permitted')
    if set(raw.columns[5:-1]) != set(targets) or not raw[targets].isin(range(7)).all().all():
        raise ValueError('Raw target domain mismatch')
    reference = published.set_index('beer')[targets]
    if not np.allclose(reference.reindex(data.meta.beer), data.Y, atol=1e-7, rtol=0):
        raise ValueError('Published S4 differs from contest labels')
    standardized = standardize_panel(raw, targets)
    discovery = sorted(targets)[:25]
    validation = sorted(targets)[25:]
    means, mapping = infer_identity(standardized, raw.beer_name, reference, discovery)
    positional = dict(zip(range(1, 251), published.beer))
    if not (mapping.beer == mapping.raw_beer.map(positional)).all() or not mapping.is_row_nearest.all():
        raise ValueError('Source order and independent profile matching do not agree')
    # A second, disjoint attribute group must recover the same identity permutation.
    _, reverse_mapping = infer_identity(standardized, raw.beer_name, reference, validation)
    if not mapping.set_index('raw_beer').beer.equals(reverse_mapping.set_index('raw_beer').beer):
        raise ValueError('Disjoint-attribute identity validation failed')
    comparison = reference.reindex(mapping.beer).to_numpy()
    reconstructed = means.reindex(mapping.raw_beer).to_numpy()
    no_constant = ~standardized.isna().any().to_numpy()
    exact_error = np.abs(reconstructed[:, no_constant] - comparison[:, no_constant])
    if exact_error.max() > cfg['mapping_tolerance']:
        raise ValueError('Nonconstant attributes do not reproduce published scores')
    mapping['published_position'] = mapping.raw_beer
    mapping['identity_status'] = 'source_order_reconstructed_and_disjoint_attribute_verified_not_author_mapping_table'
    save_csv(mapping, out / 'identity_mapping.csv')
    # Reverse engineering only: fit unknown constant replacement on odd source IDs,
    # validate on even IDs. Do not call this the documented original preprocessing.
    counts = raw.groupby('beer_name').size().reindex(mapping.raw_beer).to_numpy()
    source_ids = mapping.raw_beer.to_numpy()
    replacement_rows, error_rows = [], []
    for j, target in enumerate(targets):
        sums = standardized[target].fillna(0).groupby(raw.beer_name).sum().reindex(source_ids).to_numpy()
        missing = standardized[target].isna().groupby(raw.beer_name).sum().reindex(source_ids).to_numpy()
        if missing.sum():
            fit = (source_ids % 2 == 1) & (missing > 0)
            constant = float(np.dot(missing[fit], comparison[fit, j] * counts[fit] - sums[fit]) / np.dot(missing[fit], missing[fit]))
            rebuilt = (sums + missing * constant) / counts
            held = (source_ids % 2 == 0) & (missing > 0)
            replacement_rows.append({'target': target, 'replacement': constant, 'fit_beers': int(fit.sum()),
                                     'validation_beers': int(held.sum()),
                                     'validation_max_error': float(np.abs(rebuilt[held] - comparison[held, j]).max()),
                                     'status': 'empirically_reconstructed_not_source_documented'})
        else:
            rebuilt = reconstructed[:, j]
        error_rows.append({'target': target, 'constant_taster_attribute': bool(missing.sum()),
                           'max_error_without_fitted_replacement': float(np.abs(reconstructed[:, j] - comparison[:, j]).max()),
                           'max_error_after_conditional_reconstruction': float(np.abs(rebuilt - comparison[:, j]).max())})
    save_csv(pd.DataFrame(replacement_rows), out / 'constant_attribute_reconstruction.csv')
    save_csv(pd.DataFrame(error_rows), out / 'sensory_reconstruction_errors.csv')
    raw['beer'] = raw.beer_name.map(mapping.set_index('raw_beer').beer)
    balanced, records, person = aggregate_panel(raw, targets)
    balanced, records = balanced.reindex(data.meta.beer), records.reindex(data.meta.beer)
    save_csv(balanced.reset_index(), out / 'observed_panel_means_0_6.csv')
    save_csv(records.reset_index(), out / 'record_weighted_means_sensitivity.csv')
    n = person.groupby('beer').size().reindex(data.meta.beer)
    save_csv(pd.DataFrame({'beer': data.meta.beer.to_numpy(), 'distinct_panelists': n.to_numpy(),
                           'rating_records': raw.groupby('beer').size().reindex(data.meta.beer).to_numpy()}), out / 'panel_sample_counts.csv')
    rng = np.random.default_rng(cfg['seed'])
    precision, repeats = [], []
    for beer, group in person.groupby(level='beer'):
        a = group.to_numpy()
        draws = a[rng.integers(len(a), size=(cfg['panel_bootstrap_repeats'], len(a)))].mean(1)
        low, high = np.quantile(draws, [.025, .975], axis=0)
        for j, target in enumerate(targets):
            precision.append({'beer': int(beer), 'target': target, 'observed_mean': a[:, j].mean(),
                              'panel_bootstrap_low': low[j], 'panel_bootstrap_high': high[j], 'panelists': len(a)})
    for (beer, taster), group in raw.groupby(['beer', 'taster']):
        if group.date.nunique() < 2:
            continue
        first, last = group.sort_values('date').iloc[[0, -1]][targets].to_numpy(float)
        repeats.extend({'beer': int(beer), 'target': target, 'absolute_first_last_difference': abs(last[j] - first[j])} for j, target in enumerate(targets))
    save_csv(pd.DataFrame(precision), out / 'panel_sampling_precision.csv')
    save_csv(pd.DataFrame(repeats).groupby('target', as_index=False).agg(mean_absolute_first_last_difference=('absolute_first_last_difference', 'mean'), repeated_person_beer_pairs=('beer', 'size')), out / 'within_person_repeatability.csv')
    calibrated = pd.read_excel(external / BOOK_NAME, sheet_name='Calibrated chemical dataset').set_index('beer').reindex(data.meta.beer)
    fields = list(calibrated.columns[2:])
    chemistry, mismatches = [], []
    for field in fields:
        a = calibrated[field].to_numpy(float)
        supplied = data.X[field].to_numpy(float)
        candidates, half = transform_candidates(a)
        valid = np.isfinite(a) & np.isfinite(supplied)
        rates = {name: float(np.isclose(pred[valid], supplied[valid], atol=1e-7, rtol=1e-7).mean()) for name, pred in candidates.items()}
        best = max(rates, key=rates.get)
        good = np.isclose(candidates[best], supplied, atol=1e-7, rtol=1e-7)
        chemistry.append({'field': field, 'best_transform': best, 'match_fraction': rates[best],
                          'comparable_pairs': int(valid.sum()), 'nonpositive_raw': int((a <= 0).sum()),
                          'missing_raw': int(np.isnan(a).sum()), 'half_min_positive': half,
                          'status': 'all_comparable_pairs_reconciled' if rates[best] == 1 else 'exceptions_retained'})
        mismatches.extend({'beer': int(data.meta.iloc[i].beer), 'field': field,
                           'absolute_difference': abs(candidates[best][i] - supplied[i]),
                           'status': 'unexplained_source_difference_not_repaired'} for i in np.flatnonzero(valid & ~good))
    save_csv(pd.DataFrame(chemistry), out / 'chemical_scale_reconciliation.csv')
    save_csv(pd.DataFrame(mismatches), out / 'chemical_source_exceptions.csv')
    selected = concentration_fields(calibrated, data.thresholds, cfg['concentration_units'])
    meta = data.thresholds.dropna(subset=['field_name_R']).drop_duplicates('field_name_R').set_index('field_name_R')
    save_csv(pd.DataFrame([{'field': f, 'unit': meta.at[f, 'unit'], 'calibrated_metadata': meta.at[f, 'calibrated'],
                           'negative_raw': int((calibrated[f] < 0).sum()), 'zero_raw': int((calibrated[f] == 0).sum()),
                           'missing_raw': int(calibrated[f].isna().sum())} for f in selected]), out / 'concentration_feature_registry.csv')
    thresholds = {'DMS': (19., 'ug/L', 'attachment_median'), 'X2.3.butanedione': (.09, 'mg/L', 'attachment_median'),
                  'X4VG': (.2, 'mg/L', 'attachment_median'),
                  'X.E..2.nonenal': (cfg['literature_nonenal_threshold_mg_L'], 'mg/L', 'Meilgaard_1993_Santos_2003_external_reference_not_batch_calibration')}
    oav = []
    for field, (threshold, unit, source) in thresholds.items():
        for factor in cfg['threshold_sensitivity_multipliers']:
            for i, value in enumerate(calibrated[field]):
                oav.append({'beer': int(data.meta.iloc[i].beer), 'field': field, 'unit': unit,
                            'threshold_source': source, 'reference_threshold': threshold, 'threshold_multiplier': factor,
                            'reference_oav': value / (factor * threshold) if pd.notna(value) and value >= 0 else np.nan,
                            'measurement_status': 'positive_reported_concentration_LOD_unknown' if pd.notna(value) and value > 0 else 'zero_or_missing_not_certified_absent',
                            'sensory_target': {'DMS': 'dms', 'X2.3.butanedione': 'diacetyl', 'X4VG': 'X4vg', 'X.E..2.nonenal': 't2n'}[field]})
    save_csv(pd.DataFrame(oav), out / 'reference_oav_sensitivity.csv')
    audit_aggregate_hypotheses(data, calibrated, out)
    summary = {'raw_rows': len(raw), 'beers': len(balanced), 'panelists': int(raw.taster.nunique()), 'targets': len(targets),
               'duplicate_person_beer_records': int(raw.duplicated(['beer', 'taster']).sum()),
               'identity_matches_both_attribute_groups': len(mapping), 'nonconstant_targets_reproduced': int(no_constant.sum()),
               'nonconstant_max_error': float(exact_error.max()), 'replacement_rules_empirically_reconstructed': len(replacement_rows),
               'panelists_per_beer_min': int(n.min()), 'panelists_per_beer_max': int(n.max()),
               'mean_abs_record_vs_person_aggregation_difference': float(np.abs(balanced - records).to_numpy().mean()),
               'calibrated_reference_fields': len(fields), 'concentration_model_fields': len(selected),
               'fully_reconciled_fields': sum(row['match_fraction'] == 1 for row in chemistry),
               'unexplained_chemical_pairs': len(mismatches),
               'source_sha256': {RAW_NAME: sha256(external / RAW_NAME), BOOK_NAME: sha256(external / BOOK_NAME)},
               'mapping_scope': 'reconstructed_source_identity_not_independent_new_external_samples',
               'absolute_target_scope': cfg['primary_absolute_target'], 'physical_validation_performed': False}
    write_json(out / 'audit_summary.json', summary)
    return raw, balanced, records, calibrated[selected].where(calibrated[selected] >= 0), selected
