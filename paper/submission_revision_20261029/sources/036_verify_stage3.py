"""Cross-check saved scientific artifacts and record exact executed scope."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/stage3'
suite=ET.parse(ROOT/'tmp/stage3_full_tests.xml').getroot().find('testsuite')
tests={k:int(suite.attrib[k]) for k in ['tests','failures','errors','skipped']}
assert tests['tests']>=40 and tests['failures']==tests['errors']==tests['skipped']==0
status={part:json.loads((OUT/part/'status.json').read_text()) for part in ['q1','q2','q3']}
assert all(a['complete'] for a in status.values())
roles=pd.read_csv(OUT/'q1/outer_roles.csv')
assert len(roles)==1250 and roles['repeat'].nunique()==5
assert (roles.groupby(['repeat','beer']).size()==1).all()
grades=pd.read_csv(OUT/'q1/ASSUMPTION_oof_relative_grades.csv'); a=grades.drop(columns='beer').to_numpy()
assert a.shape==(250,50) and np.isfinite(a).all() and np.all((a>=0)&(a<=6)&(a==a.astype(int)))
weights=pd.read_csv(OUT/'q1/blend_weights.csv'); assert len(weights)==750 and weights.weight_GBR.between(0,1).all()
metadata=pd.read_csv(ROOT/'outputs/stage2/audit/source_manifest.csv')
from sys import path
path.insert(0,str(ROOT))
from src.data import find_data,load_data
original=find_data(); data=load_data(original)
for row in metadata.itertuples(): assert hashlib.sha256((original/row.file).read_bytes()).hexdigest()==row.sha256
scope=pd.read_csv(OUT/'q1/metrics_by_scope.csv'); core=[c for c in data.Y if c.startswith(('A_','F_'))]
for model in scope.model.unique():
    pred=pd.read_csv(OUT/f'q1/oof_{model}.csv').set_index('beer').loc[data.meta.beer]
    for name,fields in [('core30',core),('all50',list(data.Y))]:
        expected=float(np.abs(data.Y[fields].to_numpy()-pred[fields].to_numpy()).mean())
        actual=float(scope[(scope.model==model)&(scope.scope==name)].mae.iloc[0]); assert abs(expected-actual)<2e-8
graph=pd.read_csv(OUT/'q2/graph_fold_metrics.csv'); assert graph.graph.notna().all() and len(graph[graph.graph=='shuffled'])==300
gen=pd.read_csv(OUT/'q3/generation_comparison.csv'); assert len(gen)==60 and (gen.n_generated==40).all()
assert (gen.provided_bounds_violation_fraction==0).all()
transfer=pd.read_csv(OUT/'q3/transfer_independent_evaluation.csv'); assert len(transfer)==220
assert not transfer.absolute_chemical_certification.any() and not transfer.real_tasting_validated.any()
defects=pd.read_csv(OUT/'q3/transfer_defect_checks.csv'); assert defects.provided_nonincrease.all()
top=pd.read_csv(OUT/'q2/top3_with_evidence.csv'); assert len(top)==150 and (top.groupby('target').size()==3).all()
literature=[]
for r in top.itertuples():
    row={'target':r.target,'field':r.field,'statistical_status':r.evidence_status,'causal_status':'not_identified','independent_reference':'','chemical_evidence':'attachment_metadata_only_target_specific_mechanism_unverified','limitations':'direction_and_style_confounding_require_review'}
    if r.target=='X4vg' and r.field=='X4VG': row.update(independent_reference='https://pubmed.ncbi.nlm.nih.gov/19053838/',chemical_evidence='beer_research_supports_clove_related_aroma_not_current_dataset_causality',limitations='aging_and_matrix_conditions_not_a_new_spiking_test')
    elif r.target=='metallic' and r.field=='iron': row.update(independent_reference='https://pubmed.ncbi.nlm.nih.gov/22304665/',chemical_evidence='metal_solution_and_salivary_lipid_oxidation_mechanism',limitations='iron_speciation_and_beer_matrix_not_resolved_no_beer_threshold_inferred')
    elif r.target=='coriander' and r.field=='linalool': row.update(independent_reference='https://pubmed.ncbi.nlm.nih.gov/38398596/',chemical_evidence='coriander_seed_low_alcohol_beer_primary_research_context',limitations='not_specific_causality_for_this_panel_or_this_concentration')
    elif (r.target=='acidity' and r.field in ['lactic_acid.mg.L.','acetic_acid.g.L.']) or (r.target=='lactic' and r.field=='lactic_acid.mg.L.') or (r.target=='alcohol' and r.field=='ethanol..v.v.'):
        row.update(chemical_evidence='measurement_compatible_relation_not_independent_causal_validation',limitations='matrix_mixture_and_source_preprocessing_remain')
    literature.append(row)
pd.DataFrame(literature).to_csv(OUT/'q2/chemical_evidence_ledger.csv',index=False,encoding='utf-8-sig')
summary={'client_date':'2026-10-25','tests':tests,'original_sources_unchanged':True,'prediction_outer_folds':15,'unique_beers':250,
 'graph_null_fits_per_outer':10,'generation_disjoint_partitions':5,'generated_records_per_group':40,
 'core_targets':30,'all_targets':50,'relative_grade_cells':12500,
 'complete_absolute_score_calibration':False,'complete_chemical_certification':False,'real_tasting_validated':False,
 'scientific_scope':'exploratory_internal_validation_after_earlier_stages_not_new_external_validation',
 'executed':['nested_prediction_and_shrinkage','fixed_relative_grade_sets','missingness_and_clipping','train_only_graph_ablations','crossfit_within_style_module_effects','disjoint_generation_and_Pareto_transfer','failure_and_unknown_item_preservation'],
 'still_unresolved':['real_absolute_score_anchors','detection_limits_and_all_units','all_aggregate_formulas','full_target_specific_chemical_mechanisms','new_external_beers_and_tasting','human_independent_review'],
 'source_hashes':[{ 'file':r.file,'sha256':r.sha256} for r in metadata.itertuples()],
 'code_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').glob('stage3*.py'))}}
(OUT/'validation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
checks=[{'file':p.relative_to(OUT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='result_checksums.json' and 'private' not in p.parts]
(OUT/'result_checksums.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'tests':tests,'files_hashed':len(checks),'relative_grade_cells':12500,'all_three_experiments_complete':True},ensure_ascii=False))
