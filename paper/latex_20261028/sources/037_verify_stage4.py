"""Validate source identity, absolute outputs, grading and frozen-candidate roles."""
from pathlib import Path
import json
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'outputs/stage4'
audit = json.loads((OUT / 'audit/audit_summary.json').read_text(encoding='utf-8'))
assert audit['identity_matches_both_attribute_groups'] == 250 and audit['nonconstant_targets_reproduced'] == 38
assert audit['nonconstant_max_error'] < 1e-7 and audit['concentration_model_fields'] == 61
mapping = pd.read_csv(OUT / 'audit/identity_mapping.csv'); assert mapping.beer.nunique() == 250 and mapping.raw_beer.nunique() == 250 and mapping.is_row_nearest.all()
reconstruction = pd.read_csv(OUT / 'audit/constant_attribute_reconstruction.csv'); assert len(reconstruction) == 12 and reconstruction.validation_max_error.max() < 1e-7
panel = pd.read_csv(OUT / 'audit/observed_panel_means_0_6.csv').set_index('beer'); assert panel.shape == (250, 50) and panel.ge(0).all().all() and panel.le(6).all().all()
for part in ['prediction', 'ordinal', 'transfer']: assert json.loads((OUT / part / 'status.json').read_text())['complete']
graph_status = json.loads((OUT / 'raw_graph/q2/status.json').read_text()); assert graph_status['complete'] and graph_status['outer_folds'] == 15
graph_folds = pd.read_csv(OUT / 'raw_graph/q2/graph_fold_metrics.csv')
assert len(graph_folds[graph_folds.graph == 'shuffled']) == 300
roles = pd.read_csv(OUT / 'prediction/outer_roles.csv'); assert len(roles) == 1250 and (roles.groupby(['repeat', 'beer']).size() == 1).all()
metrics = pd.read_csv(OUT / 'prediction/metrics_by_scope.csv')
core = [f for f in panel if f.startswith(('A_', 'F_'))]
for model in metrics.model.unique():
    p = pd.read_csv(OUT / f'prediction/oof_{model}_0_6.csv').set_index('beer').reindex(panel.index)
    assert p.ge(0).all().all() and p.le(6).all().all()
    for scope, fields in [('core30', core), ('all50', list(panel))]:
        value = np.abs(panel[fields] - p[fields]).to_numpy().mean()
        assert abs(value - metrics[(metrics.model == model) & (metrics.scope == scope)].mae.iloc[0]) < 1e-7
for name in ['oof_modal_grades_0_6.csv', 'oof_median_grades_0_6.csv', 'oof_rounded_panel_mean_grades_0_6.csv']:
    grades = pd.read_csv(OUT / 'ordinal' / name).iloc[:, 1:].to_numpy(); assert grades.shape == (250, 50) and np.isin(grades, np.arange(7)).all()
distributions = pd.read_csv(OUT / 'ordinal/oof_rating_distributions.csv'); assert len(distributions) == 12500
prob = distributions[[f'predicted_p{k}' for k in range(7)]].to_numpy(); observed = distributions[[f'observed_panel_p{k}' for k in range(7)]].to_numpy()
assert (prob >= 0).all() and np.allclose(prob.sum(1), 1, atol=1e-8) and np.allclose(observed.sum(1), 1, atol=1e-8)
assert np.array_equal(prob.argmax(1), distributions.predicted_modal_grade)
assert abs(np.abs((prob * np.arange(7)).sum(1) - (observed * np.arange(7)).sum(1)).mean() - pd.read_csv(OUT / 'ordinal/metrics_by_scope.csv').query("scope == 'all50' and method == 'ET_rating_distribution'").expected_panel_mean_MAE.iloc[0]) < 1e-7
intervals = pd.read_csv(OUT / 'prediction/absolute_panel_mean_intervals.csv'); assert len(intervals) == 12500 and (intervals.lower_0_6 <= intervals.upper_0_6).all() and intervals.lower_0_6.ge(0).all() and intervals.upper_0_6.le(6).all()
top = pd.read_csv(OUT / 'prediction/absolute_raw_top3.csv'); assert len(top) == 150 and (top.groupby('target').size() == 3).all() and not top.causal_validated.any()
frozen = pd.read_csv(OUT / 'transfer/frozen_stage3_candidates_raw_score_check.csv'); assert len(frozen) == 220 and not frozen.physical_validation.any()
assert not (frozen.original_and_raw_proxy_support & ~frozen.original_proxy_acceptance).any()
mixture = pd.read_csv(OUT / 'transfer/concentration_mixture_evaluation.csv'); assert len(mixture) == 55 and not mixture.real_tasting_validated.any() and not mixture.LOD_certification.any()
assert all(json.loads((OUT / f'transfer/concentration_candidate_lock_{i}.json').read_text())['selection_completed_before_J_fitting'] for i in range(5))
assert pd.read_csv(OUT / 'protocol/blank_panel_rating_template.csv').empty
tests = ET.parse(ROOT / 'tmp/stage4_tests.xml').getroot(); suites = list(tests.iter('testsuite'))
test_summary = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ['tests', 'failures', 'errors', 'skipped']}
assert test_summary['tests'] == 48 and not any(test_summary[k] for k in ['failures', 'errors', 'skipped'])
summary = {'all_four_stage4_components_complete': True, 'tests': test_summary, 'identity_mapping_verified': True,
           'raw_endpoint_graph_controls_complete': True,
           'absolute_panel_mean_cells': 12500, 'integer_grade_cells_per_rule': 12500, 'ordinal_probability_rows': 12500,
           'frozen_transfer_records': len(frozen), 'concentration_mixture_records': len(mixture), 'physical_trial_observations': 0}
(OUT / 'validation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
hashes = {p.relative_to(OUT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'result_checksums.json'}
(OUT / 'result_checksums.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({**summary, 'hashed_files': len(hashes)}, ensure_ascii=False))
