"""Auditable field-set, classification-metric and candidate evidence checks."""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import f1_score,balanced_accuracy_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data import load_data,find_data,AGGREGATES
from src.common import save_csv,write_json,sha256

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-dir');ap.add_argument('--out',default='outputs/submission_review_20261029');a=ap.parse_args();out=ROOT/a.out
 if out.exists() and any(out.iterdir()):raise SystemExit('choose an empty output directory')
 out.mkdir(parents=True,exist_ok=True);D=load_data(find_data(a.data_dir));beers=D.meta.beer.to_numpy();styles=D.meta.tasting_category_fine.to_numpy();targets=list(D.Y);a4=ROOT/'outputs/stage4/audit';rec=pd.read_csv(a4/'chemical_scale_reconciliation.csv');reg=pd.read_csv(a4/'concentration_feature_registry.csv');exc=pd.read_csv(a4/'chemical_source_exceptions.csv');manifest=json.loads((ROOT/'outputs/joint_20261028/manifest.json').read_text(encoding='utf8'));selected=set(manifest['verified_fields']);equal=set(rec.loc[rec.status=='all_comparable_pairs_reconciled','field']);diff=set(rec.loc[rec.status=='exceptions_retained','field']);unit=set(reg.field);nonagg=set(D.X)-AGGREGATES;computed=equal & unit & nonagg
 assert len(equal)==38 and len(diff)==32 and len(exc)==80 and len(computed)==32 and selected==computed and not selected&diff
 rows=[]
 for r in rec.itertuples():rows.append({'field':r.field,'audit_status':r.status,'best_transform':r.best_transform,'comparable_pairs':r.comparable_pairs,'exception_pairs':int((exc.field==r.field).sum()),'documented_unit_in_registry':r.field in unit,'nonaggregate_model_field':r.field in nonagg,'selected_for_32_subset':r.field in selected,'unit':reg.set_index('field').unit.get(r.field,'')})
 save_csv(pd.DataFrame(rows),out/'field_sets.csv')
 x=D.X['diethyl.succinate'].to_numpy(); raw=D.calibrated['diethyl.succinate'].to_numpy();mask=np.isfinite(x)&np.isfinite(raw)&(raw>0);r=10**x[mask]/raw[mask]
 identity={'field':'diethyl.succinate','comparable_positive_pairs':int(mask.sum()),'identity_max_absolute_difference':float(np.max(abs(x[mask]-raw[mask]))),'counterfactual_exponentiation_ratio_min_median_max':[float(np.min(r)),float(np.median(r)),float(np.max(r))],'counterfactual_ratio_note':'10^X/reference is a forced log hypothesis applied to identity-aligned values; variation is not evidence of a constant unit conversion or actual quantification bias','independent_chemical_calibration':False}
 write_json(out/'identity_field_check.json',identity)
 ratings=pd.read_csv(ROOT/'outputs/stage4/ordinal/oof_rating_distributions.csv').set_index(['beer','target']).loc[pd.MultiIndex.from_product([beers,targets])];q=ratings[[f'observed_panel_p{k}' for k in range(7)]].to_numpy().reshape(len(beers),50,7);q/=q.sum(-1,keepdims=True);prob=ratings[[f'predicted_p{k}' for k in range(7)]].to_numpy().reshape(q.shape);prob/=prob.sum(-1,keepdims=True);roles=pd.read_csv(ROOT/'outputs/stage4/prediction/outer_roles.csv');preds={'frequency_ET':prob,'population':np.zeros_like(q),'style_only':np.zeros_like(q)};count=np.zeros(len(beers))
 for fold in sorted(roles.fold.unique()):
  test=np.flatnonzero(np.isin(beers,roles.loc[roles.fold==fold,'beer']));train=np.setdiff1d(np.arange(len(beers)),test);preds['population'][test]+=q[train].mean(0)
  for i in test:
   ids=train[styles[train]==styles[i]]; preds['style_only'][i]+=q[ids].mean(0) if len(ids) else q[train].mean(0)
  count[test]+=1
 assert np.all(count==5)
 for name in ['population','style_only']:preds[name]/=count[:,None,None]
 observed=q.argmax(-1);scopes={'core30':[i for i,t in enumerate(targets) if t.startswith(('A_','F_'))],'all50':list(range(50))};metrics=[];targetrows=[];confusions=[]
 for name,p in preds.items():
  grade=p.argmax(-1)
  for scope,cols in scopes.items():
   y=observed[:,cols].ravel();z=grade[:,cols].ravel();counts=np.bincount(y,minlength=7);active=np.flatnonzero(counts>0); nz=y>0;cm=confusion_matrix(y,z,labels=np.arange(7));
   metrics.append({'model':name,'scope':scope,'pooled_records':len(y),'zero_fraction':float((y==0).mean()),'pooled_macro_f1_7_observed_grades':float(f1_score(y,z,labels=active,average='macro',zero_division=0)),'pooled_balanced_accuracy':float(balanced_accuracy_score(y,z)),'modal_grade_mae':float(np.abs(y-z).mean()),'exact_accuracy':float((y==z).mean()),'nonzero_exact_accuracy':float((z[nz]==y[nz]).mean()),'nonzero_detection_recall':float((z[nz]>0).mean())})
   for i in range(7):
    for j in range(7):confusions.append({'model':name,'scope':scope,'observed_grade':i,'predicted_grade':j,'count':int(cm[i,j])})
  for j,t in enumerate(targets):
   y=observed[:,j];z=grade[:,j];active=np.unique(y);targetrows.append({'model':name,'target':t,'n':len(y),'observed_nonzero':int((y>0).sum()),'observed_grades':';'.join(str(k) for k in active),'macro_f1_over_observed_grades':float(f1_score(y,z,labels=active,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,z)),'modal_grade_mae':float(np.abs(y-z).mean())})
 save_csv(pd.DataFrame(metrics),out/'ordinal_metrics.csv');save_csv(pd.DataFrame(targetrows),out/'ordinal_metrics_by_target.csv');save_csv(pd.DataFrame(confusions),out/'ordinal_confusion.csv')
 evidence=pd.read_csv(ROOT/'outputs/stage4/raw_graph/q2/top3_with_evidence.csv');assert len(evidence)==150
 support=(evidence.bootstrap_frequency_conditional_on_full_selected_params>=.5)&(evidence.crossfit_selection_count>=5)&(evidence.heldout_delta_on_selected_folds>0)&(evidence.within_style_delta_on_selected_folds>0);assert support.sum()==36;evidence['support_class']=np.where(support,'supported_descriptive','exploratory');save_csv(evidence,out/'top3_classified.csv');group=evidence.groupby('target').support_class.apply(lambda z:(z=='supported_descriptive').sum());save_csv(group.reset_index(name='supported_count'),out/'top3_support_by_target.csv')
 write_json(out/'manifest.json',{'source_files':{str(p.relative_to(ROOT)):sha256(p) for p in [a4/'chemical_scale_reconciliation.csv',a4/'concentration_feature_registry.csv',a4/'chemical_source_exceptions.csv',ROOT/'outputs/joint_20261028/manifest.json',ROOT/'outputs/stage4/ordinal/oof_rating_distributions.csv',ROOT/'outputs/stage4/prediction/outer_roles.csv',ROOT/'outputs/stage4/raw_graph/q2/top3_with_evidence.csv']},'script_sha256':sha256(Path(__file__)),'32_unit_subset_and_32_exception_overlap':len(selected&diff),'metric_scope':'pooled beer-attribute modal grades, macro across observed grades; repeated OOF probabilities averaged per beer before decisions; 7 grades are ordered; no independence claim','classification_thresholds':'descriptive exploratory evidence thresholds; selection not independent confirmatory test','no_physical_validation':True})
 print('32 verified subset vs exceptions overlap',len(selected&diff));print('identity',identity);print(pd.DataFrame(metrics).to_string(index=False));print('supported by target',group.value_counts().sort_index().to_string())
if __name__=='__main__':main()
